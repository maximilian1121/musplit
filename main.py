import os
import json
import hashlib
import asyncio
import shutil
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from audio_separator.separator import Separator
import yt_dlp

app = FastAPI()

HISTORY_FILE = "history.json"
OUTPUT_DIR = "separated_audio"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

split_lock = asyncio.Lock()
history_lock = asyncio.Lock()


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


def sanitize_filename(name):
    """Removes characters that are illegal in file systems."""
    return re.sub(r'[\\/*?:"<>|]', "", name)


# ------------------------
# Separator setup
# ------------------------
sep = Separator(output_dir=OUTPUT_DIR, model_file_dir=MODEL_DIR)
current_model = None
AVAILABLE_MODELS = sep.get_simplified_model_list()


# ------------------------
# Core processing
# ------------------------
async def process_task(url: str, task_hash: str, model_name: str):
    try:
        base_path = Path(OUTPUT_DIR).resolve()
        task_folder = base_path / task_hash
        task_folder.mkdir(parents=True, exist_ok=True)

        # 1. Fetch metadata first to get the title
        yield f"data: {json.dumps({'status': 'Fetching metadata...'})}\n\n"

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_title = info.get("title", "Unknown_Track")
            clean_title = sanitize_filename(raw_title)

        # Path for the permanent cached audio
        wav_path = task_folder / f"{clean_title}.wav"

        # 2. Check Cache / Download
        if wav_path.exists():
            yield f"data: {json.dumps({'status': 'Using cached audio', 'title': clean_title})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'Downloading...'})}\n\n"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(task_folder / f"{clean_title}.%(ext)s"),
                "quiet": True,
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)

        # 3. Separation
        async with split_lock:
            global current_model
            if model_name != current_model:
                yield f"data: {json.dumps({'status': f'Loading {model_name}'})}\n\n"
                sep.load_model(model_name)
                current_model = model_name

            yield f"data: {json.dumps({'status': 'Separating audio'})}\n\n"

            stems_temp_dir = task_folder / "stems_temp"
            stems_temp_dir.mkdir(exist_ok=True)

            sep.output_dir = str(stems_temp_dir)
            # The library typically prefixes the output with the input filename
            output_files = sep.separate(str(wav_path))

            # Manual move fix
            for file_name in output_files:
                stray_file = base_path / file_name
                target_file = stems_temp_dir / file_name
                if stray_file.exists() and not target_file.exists():
                    shutil.move(str(stray_file), str(target_file))

            # Copy the original titled wav into the zip
            shutil.copy(
                str(wav_path), str(stems_temp_dir / f"{clean_title}_Original.wav")
            )

            # 4. Zip the temp folder
            yield f"data: {json.dumps({'status': 'Zipping...'})}\n\n"
            zip_filename = f"{clean_title}_{task_hash}_stems"
            shutil.make_archive(
                str(base_path / zip_filename), "zip", str(stems_temp_dir)
            )

            shutil.rmtree(stems_temp_dir)

        # 5. History update
        async with history_lock:
            h = get_history()
            h[task_hash] = {
                "url": url,
                "title": clean_title,
                "zip": f"{zip_filename}.zip",
                "model": model_name,
                "status": "complete",
            }
            save_history(h)

        yield f"data: {json.dumps({'status': 'Done!', 'complete': True, 'zip': f'{zip_filename}.zip'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'status': f'Error: {str(e)}', 'error': True})}\n\n"


# ------------------------
# API Endpoints
# ------------------------
@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/models")
async def models_api():
    return AVAILABLE_MODELS


@app.get("/history")
async def history_api():
    return get_history()


@app.get("/download/{filename}")
async def download(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=safe_name)
    return {"error": "File not found."}


@app.get("/separate")
async def separate_api(url: str, model: str):
    if model not in AVAILABLE_MODELS:
        return {"error": "Invalid model name."}

    task_hash = hashlib.md5((url + model).encode()).hexdigest()[:10]

    return StreamingResponse(
        process_task(url, task_hash, model), media_type="text/event-stream"
    )
