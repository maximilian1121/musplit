import os, json, hashlib, asyncio, shutil
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from audio_separator.separator import Separator
import yt_dlp

app = FastAPI()
HISTORY_FILE = "history.json"
OUTPUT_DIR = "separated_audio"
MODEL_DIR = "models"
split_lock = asyncio.Lock()

for d in [OUTPUT_DIR, MODEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


async def process_task(url: str, task_hash: str):
    history = get_history()
    if task_hash in history and history[task_hash].get("status") == "complete":
        yield f"data: {json.dumps({'status': 'Already exists!', 'complete': True, 'hash': task_hash})}\n\n"
        return

    try:
        # 1. Download & Convert to WAV (Fixes the webm complaint)
        yield f"data: {json.dumps({'status': 'Downloading & Converting...', 'hash': task_hash})}\n\n"
        temp_input = os.path.join(OUTPUT_DIR, f"{task_hash}_raw")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_input,
            "quiet": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown Track")
            input_wav = f"{temp_input}.wav"

        # 2. Queue for AI
        yield f"data: {json.dumps({'status': 'Waiting in Queue...', 'hash': task_hash})}\n\n"

        async with split_lock:
            yield f"data: {json.dumps({'status': 'Splitting (UVR-MDX)...', 'hash': task_hash})}\n\n"

            task_folder = os.path.join(OUTPUT_DIR, task_hash)
            if not os.path.exists(task_folder):
                os.makedirs(task_folder)

            sep = Separator(output_dir=task_folder, model_file_dir=MODEL_DIR)
            sep.load_model("UVR-MDX-NET-Voc_FT.onnx")
            output_files = sep.separate(input_wav)

            # Create ZIP
            zip_name = f"{task_hash}_stems"
            shutil.make_archive(os.path.join(OUTPUT_DIR, zip_name), "zip", task_folder)

        # 3. Cleanup & Save
        if os.path.exists(input_wav):
            os.remove(input_wav)
        history[task_hash] = {
            "url": url,
            "status": "complete",
            "title": title,
            "zip": f"{zip_name}.zip",
        }
        save_history(history)
        yield f"data: {json.dumps({'status': 'Finished!', 'hash': task_hash, 'complete': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'status': f'Error: {str(e)}', 'hash': task_hash, 'error': True})}\n\n"


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/history")
async def history_api():
    return get_history()


@app.get("/download/{filename}")
async def download(filename: str):
    # Basic path traversal protection
    safe_name = os.path.basename(filename)
    return FileResponse(os.path.join(OUTPUT_DIR, safe_name), filename=safe_name)


@app.get("/separate")
async def separate_endpoint(url: str):
    task_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    return StreamingResponse(
        process_task(url, task_hash), media_type="text/event-stream"
    )
