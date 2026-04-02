from __future__ import annotations

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from common_monitoring import build_rtsp_url, ensure_models_ready, mjpeg_frame_bytes, run_single_frame_analysis, startup_models, state, test_rtsp_connection


app = FastAPI(title="Safety Monitoring API - RTSP", description="RTSP 카메라 기반 안전 모니터링 API", version="1.0.0")


class CameraRegisterRequest(BaseModel):
    ip_address: str = Field(..., description="카메라 IP 주소", example="192.168.0.50")
    camera_id: str = Field(..., description="카메라 계정 ID", example="admin")
    camera_pw: str = Field(..., description="카메라 계정 비밀번호", example="1234")
    rtsp_port: int = Field(554, description="RTSP 포트", example=554)
    rtsp_path: str = Field("/stream1", description="RTSP 경로", example="/stream1")


class CameraRegisterResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="RTSP 연결 성공")


class CameraHealthResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="카메라 연결 정상")

class InternalVLMAnalysisRequest(BaseModel):
    camera_ip: str = Field(..., example="192.168.0.40")
    ev_code_name: str = Field(..., example="FALL_DETECTED")
    risk_text: str = Field(..., example="1구역 cam-01에서 낙상 위험이 감지되었습니다.")
    time: str = Field(..., example="2026-03-31T22:10:00+09:00")


@app.on_event("startup")
def startup_event():
    startup_models()


def cam_stream_generator():
    ensure_models_ready()
    with state.camera_state_lock:
        if not state.camera_state["registered"]:
            raise RuntimeError("등록된 카메라가 없습니다.")

        camera_ip = state.camera_state["ip_address"]
        rtsp_url = build_rtsp_url(
            ip_address=state.camera_state["ip_address"],
            camera_id=state.camera_state["camera_id"],
            camera_pw=state.camera_state["camera_pw"],
            rtsp_port=state.camera_state["rtsp_port"],
            rtsp_path=state.camera_state["rtsp_path"],
        )
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("RTSP 스트림을 열 수 없습니다.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = run_single_frame_analysis(
                frame,
                source_name="camera",
                camera_ip=camera_ip,
            )
            display_frame = cv2.imread(result["frame_path"])
            if display_frame is None:
                continue
            chunk = mjpeg_frame_bytes(display_frame)
            if chunk is not None:
                yield chunk
    finally:
        cap.release()


@app.post("/vlm/api/camera/register", response_model=CameraRegisterResponse, tags=["camera"])
def register_camera(req: CameraRegisterRequest):
    rtsp_url = build_rtsp_url(req.ip_address, req.camera_id, req.camera_pw, req.rtsp_port, req.rtsp_path)
    success = test_rtsp_connection(rtsp_url)
    with state.camera_state_lock:
        state.camera_state["registered"] = True
        state.camera_state["ip_address"] = req.ip_address
        state.camera_state["camera_id"] = req.camera_id
        state.camera_state["camera_pw"] = req.camera_pw
        state.camera_state["rtsp_port"] = req.rtsp_port
        state.camera_state["rtsp_path"] = req.rtsp_path
        state.camera_state["last_health"] = success
    if success:
        return {"status": "success", "message": "RTSP 연결 성공"}
    return {"status": "fail", "message": "RTSP 연결 실패"}


@app.get("/vlm/api/camera/health", response_model=CameraHealthResponse, tags=["camera"])
def get_camera_health():
    with state.camera_state_lock:
        if not state.camera_state["registered"]:
            raise HTTPException(status_code=404, detail="등록된 카메라가 없습니다.")
        rtsp_url = build_rtsp_url(
            state.camera_state["ip_address"],
            state.camera_state["camera_id"],
            state.camera_state["camera_pw"],
            state.camera_state["rtsp_port"],
            state.camera_state["rtsp_path"],
        )
    success = test_rtsp_connection(rtsp_url)
    with state.camera_state_lock:
        state.camera_state["last_health"] = success
    if success:
        return {"status": "success", "message": "카메라 연결 정상"}
    return {"status": "fail", "message": "카메라 연결 끊김 또는 RTSP 접속 실패"}


@app.get("/vlm/api/model/health")
def get_model_status():
    with state.model_lock:
        return state.model_status


@app.get("/risk/latest")
def get_latest_risk():
    with state.latest_risk_lock:
        return state.latest_risk


@app.get("/frame/latest")
def get_latest_frame():
    path = state.video_state.get("last_frame_path")
    if not path:
        raise HTTPException(status_code=404, detail="아직 생성된 분석 프레임이 없습니다.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/cam", response_class=HTMLResponse)
def camera_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>RTSP Camera Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; background: #111; color: #eee; text-align: center; }
            img { max-width: 95vw; border: 2px solid #444; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>RTSP YOLO 분석 화면</h1>
        <img src="/cam/stream" alt="camera stream" />
    </body>
    </html>
    '''


@app.get("/cam/stream")
def camera_stream():
    with state.camera_state_lock:
        if not state.camera_state["registered"]:
            raise HTTPException(status_code=404, detail="등록된 카메라가 없습니다.")
    return StreamingResponse(cam_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/internal/vlm-analysis", tags=["internal"])
def receive_internal_vlm_analysis(req: InternalVLMAnalysisRequest):
    with state.internal_vlm_analysis_lock:
        state.internal_vlm_analysis = req.dict()
    return {"status": "ok"}

@app.get("/api/internal/vlm-analysis/latest", tags=["internal"])
def get_internal_vlm_analysis_latest():
    with state.internal_vlm_analysis_lock:
        return state.internal_vlm_analysis or {}