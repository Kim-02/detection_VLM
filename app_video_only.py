from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from common_monitoring import (
    VIDEO_DIR,
    ensure_models_ready,
    mjpeg_frame_bytes,
    run_single_frame_analysis,
    startup_models,
    state,
)

app = FastAPI(
    title="Safety Monitoring API - Video",
    description="동영상 파일 기반 안전 모니터링 API",
    version="1.0.0",
)


class VideoStartRequest(BaseModel):
    video_name: str


class InternalVLMAnalysisRequest(BaseModel):
    ip_addr: str
    ev_code_name: str
    risk_text: str
    time: str


@app.on_event("startup")
def startup_event():
    startup_models()


def video_worker(video_path: Path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"동영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_interval = 1.0 / fps

        with state.video_state_lock:
            state.video_state["running"] = True
            state.video_state["video_name"] = video_path.name
            state.video_state["last_error"] = None

        while not state.video_stop_event.is_set():
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            run_single_frame_analysis(frame, source_name=video_path.name)

            elapsed = time.time() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        cap.release()

    except Exception as e:
        with state.video_state_lock:
            state.video_state["last_error"] = str(e)

    finally:
        with state.video_state_lock:
            state.video_state["running"] = False

        state.video_stop_event.clear()
        state.video_worker_thread = None


def video_stream_generator():
    while True:
        with state.video_state_lock:
            running = state.video_state["running"]

        with state.stream_frame_lock:
            frame = None if state.latest_stream_frame is None else state.latest_stream_frame.copy()

        if frame is not None:
            chunk = mjpeg_frame_bytes(frame)
            if chunk is not None:
                yield chunk

        if not running:
            time.sleep(0.2)
        else:
            time.sleep(0.03)


@app.get("/vlm/api/model/health")
def get_model_status():
    with state.model_lock:
        return state.model_status


@app.get("/status/video")
def get_video_status():
    with state.video_state_lock:
        return state.video_state


@app.get("/risk/latest")
def get_latest_risk():
    with state.latest_risk_lock:
        return state.latest_risk


@app.get("/frame/latest")
def get_latest_frame():
    with state.video_state_lock:
        last_frame_path = state.video_state.get("last_frame_path")

    if not last_frame_path:
        raise HTTPException(status_code=404, detail="아직 생성된 분석 프레임이 없습니다.")

    return FileResponse(last_frame_path, media_type="image/jpeg")


@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile = File(...)):
    ensure_models_ready()

    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩에 실패했습니다.")

    result = run_single_frame_analysis(frame, source_name="upload")
    return FileResponse(result["frame_path"], media_type="image/jpeg")


@app.post("/video/start")
def start_video_analysis(req: VideoStartRequest):
    ensure_models_ready()

    video_path = VIDEO_DIR / req.video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"동영상 파일이 없습니다: {video_path}")

    with state.video_state_lock:
        if state.video_state["running"]:
            raise HTTPException(status_code=409, detail="이미 다른 동영상 분석이 실행 중입니다.")

    state.video_stop_event.clear()
    state.video_worker_thread = threading.Thread(
        target=video_worker,
        args=(video_path,),
        daemon=True,
    )
    state.video_worker_thread.start()

    return {
        "message": "동영상 분석을 시작했습니다.",
        "video_name": req.video_name,
    }


@app.post("/video/stop")
def stop_video_analysis():
    with state.video_state_lock:
        if not state.video_state["running"]:
            return {"message": "현재 실행 중인 동영상 분석이 없습니다."}

    state.video_stop_event.set()
    return {"message": "동영상 분석 중지 요청을 보냈습니다."}


@app.post("/api/internal/vlm-analysis")
def receive_internal_vlm_analysis(req: InternalVLMAnalysisRequest):
    with state.internal_vlm_analysis_lock:
        state.internal_vlm_analysis = req.dict()
    return {"status": "ok"}


@app.get("/api/internal/vlm-analysis/latest")
def get_internal_vlm_analysis_latest():
    with state.internal_vlm_analysis_lock:
        return state.internal_vlm_analysis or {}


@app.get("/cam", response_class=HTMLResponse)
def camera_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video YOLO Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; background: #111; color: #eee; text-align: center; }
            img { max-width: 95vw; border: 2px solid #444; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>동영상 YOLO 분석 화면</h1>
        <img src="/cam/stream" alt="video stream" />
    </body>
    </html>
    """


@app.get("/cam/stream")
def camera_stream():
    return StreamingResponse(
        video_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )