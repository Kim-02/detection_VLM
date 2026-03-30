from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

import qwen_tensorrt as tensorrt
import yolo_detection


app = FastAPI(
    title="Safety Monitoring API",
    description="YOLO + TensorRT Qwen 기반 건설 현장 안전 모니터링 API",
    version="1.0.0",
)


# =========================
# 전역 상태
# =========================
model_status = {"model": "loading"}
model_lock = threading.Lock()

video_worker_thread: Optional[threading.Thread] = None
video_stop_event = threading.Event()
video_state_lock = threading.Lock()
video_state = {
    "running": False,
    "video_name": None,
    "last_frame_path": None,
    "last_frame_updated_at": None,
    "last_error": None,
}

latest_risk_lock = threading.Lock()
latest_risk = {
    "risk_text": None,
    "updated_at": None,
    "video_name": None,
    "analysis": None,
    "detections": None,
}


yolo_model: Optional[YOLO] = None
qwen_runner: Optional[tensorrt.TensorRTQwenRunner] = None


# =========================
# 경로 설정
# =========================
BASE_DIR = Path.home() / "detection_VLM"
VIDEO_DIR = BASE_DIR
RUNTIME_DIR = Path.home() / "edgellm_work" / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
LAST_FRAME_PATH = RUNTIME_DIR / "latest_frame.jpg"


# =========================
# 유틸
# =========================
def resize_to_640(frame):
    return cv2.resize(frame, (640, 640))


def draw_detections(frame, detections):
    output = frame.copy()

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        class_name = det["class_name"]
        conf = det["conf"]

        label = f"{class_name} {conf:.2f}"

        color = (0, 255, 0)
        if class_name.lower() == "fire":
            color = (0, 0, 255)
        elif class_name.lower() == "smoke":
            color = (0, 165, 255)
        elif class_name.lower() == "person":
            color = (255, 0, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return output


def draw_status(frame, analysis):
    output = frame.copy()

    text = (
        f"person: {analysis['person_count']}  "
        f"fire: {'yes' if analysis['has_fire'] else 'no'}  "
        f"smoke: {'yes' if analysis['has_smoke'] else 'no'}"
    )

    color = (255, 255, 255)
    if analysis["has_fire"] or analysis["has_smoke"]:
        color = (0, 0, 255)

    cv2.putText(
        output,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return output


def analyze_detected_classes(detections):
    person_count = 0
    has_fire = False
    has_smoke = False

    for det in detections:
        class_name = det.get("class_name", "").lower()

        if class_name == "person":
            person_count += 1
        elif class_name == "fire":
            has_fire = True
        elif class_name == "smoke":
            has_smoke = True

    return {
        "person_count": person_count,
        "has_fire": has_fire,
        "has_smoke": has_smoke,
    }


def build_vlm_prompt(detections, analysis):
    detection_lines = []
    for i, det in enumerate(detections):
        detection_lines.append(
            f"[{i}] class={det['class_name']} conf={det['conf']:.2f} "
            f"box=({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])})"
        )

    detection_text = "\n".join(detection_lines) if detection_lines else "탐지 결과 없음"

    return f"""
당신은 건설 현장 안전 모니터링 도우미입니다.
반드시 한국어로만 답변하세요.

탐지 요약:
- 사람 수: {analysis['person_count']}
- 화재 여부: {'예' if analysis['has_fire'] else '아니오'}
- 연기 여부: {'예' if analysis['has_smoke'] else '아니오'}

YOLO 탐지 결과:
{detection_text}

출력 형식:
위험상황: <짧게 1~2문장>

규칙:
- 이미지를 가장 우선해서 판단하세요.
- YOLO 탐지 결과는 보조 정보로만 사용하세요.
- 위험한 상황이면 무엇이 위험한지 짧고 분명하게 설명하세요.
- 위험하지 않으면 현재 상황을 짧게 설명하세요.
- 불꽃이나 연기가 보이면 반드시 언급하세요.
- 같은 내용을 반복하지 마세요.
- 길게 쓰지 마세요.
""".strip()


def run_single_frame_analysis(frame, video_name: Optional[str] = None):
    global yolo_model, qwen_runner

    if yolo_model is None or qwen_runner is None:
        raise RuntimeError("모델이 아직 로드되지 않았습니다.")

    resize_frame = resize_to_640(frame)
    detections = yolo_detection.detect_positions_with_class_on_frame(yolo_model, resize_frame)
    analysis = analyze_detected_classes(detections)

    display_frame = draw_detections(resize_frame, detections)
    display_frame = draw_status(display_frame, analysis)

    cv2.imwrite(str(LAST_FRAME_PATH), display_frame)

    risk_text = None
    if analysis["has_fire"] or analysis["has_smoke"]:
        rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        prompt = build_vlm_prompt(detections, analysis)
        risk_text = qwen_runner.infer(
            image_input=pil_image,
            user_text=prompt,
            max_new_tokens=64,
        )

        with latest_risk_lock:
            latest_risk["risk_text"] = risk_text
            latest_risk["updated_at"] = int(time.time() * 1000)
            latest_risk["video_name"] = video_name
            latest_risk["analysis"] = analysis
            latest_risk["detections"] = detections

    with video_state_lock:
        video_state["last_frame_path"] = str(LAST_FRAME_PATH)
        video_state["last_frame_updated_at"] = int(time.time() * 1000)

    return {
        "detections": detections,
        "analysis": analysis,
        "risk_text": risk_text,
        "frame_path": str(LAST_FRAME_PATH),
    }


def video_worker(video_path: Path):
    global video_worker_thread

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"동영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_interval = 1.0 / fps
        last_vlm_trigger_time = 0.0
        vlm_cooldown = 5.0

        with video_state_lock:
            video_state["running"] = True
            video_state["video_name"] = video_path.name
            video_state["last_error"] = None

        while not video_stop_event.is_set():
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            resize_frame = resize_to_640(frame)
            detections = yolo_detection.detect_positions_with_class_on_frame(yolo_model, resize_frame)
            analysis = analyze_detected_classes(detections)

            display_frame = draw_detections(resize_frame, detections)
            display_frame = draw_status(display_frame, analysis)
            cv2.imwrite(str(LAST_FRAME_PATH), display_frame)

            with video_state_lock:
                video_state["last_frame_path"] = str(LAST_FRAME_PATH)
                video_state["last_frame_updated_at"] = int(time.time() * 1000)

            if analysis["has_fire"] or analysis["has_smoke"]:
                now = time.time()
                if now - last_vlm_trigger_time >= vlm_cooldown:
                    rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    prompt = build_vlm_prompt(detections, analysis)
                    risk_text = qwen_runner.infer(
                        image_input=pil_image,
                        user_text=prompt,
                        max_new_tokens=64,
                    )

                    with latest_risk_lock:
                        latest_risk["risk_text"] = risk_text
                        latest_risk["updated_at"] = int(time.time() * 1000)
                        latest_risk["video_name"] = video_path.name
                        latest_risk["analysis"] = analysis
                        latest_risk["detections"] = detections

                    last_vlm_trigger_time = now

            elapsed = time.time() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        cap.release()

    except Exception as e:
        with video_state_lock:
            video_state["last_error"] = str(e)
    finally:
        with video_state_lock:
            video_state["running"] = False
        video_stop_event.clear()
        video_worker_thread = None


# =========================
# 요청/응답 모델
# =========================
class VideoStartRequest(BaseModel):
    video_name: str


@app.on_event("startup")
def startup_event():
    global yolo_model, qwen_runner

    try:
        yolo_model = YOLO(str(BASE_DIR / "best.engine"))
        qwen_runner = tensorrt.TensorRTQwenRunner(
            engine_dir="~/edgellm_work/engines/qwen3-vl-2b",
            multimodal_engine_dir="~/edgellm_work/visual_engines/qwen3-vl-2b",
            llm_inference_bin="~/TensorRT-Edge-LLM/build/examples/llm/llm_inference",
            plugin_path="~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so",
            work_dir="~/edgellm_work/runtime",
        )
        with model_lock:
            model_status["model"] = "ok"
    except Exception as e:
        with model_lock:
            model_status["model"] = f"error: {e}"


@app.get("/status/model", summary="모델 상태 확인")
def get_model_status():
    with model_lock:
        return model_status


@app.get("/status/video", summary="동영상 처리 상태 확인")
def get_video_status():
    with video_state_lock:
        return video_state


@app.get("/risk/latest", summary="최신 위험 분석 결과 조회")
def get_latest_risk():
    with latest_risk_lock:
        return latest_risk


@app.get("/frame/latest", summary="최신 YOLO 분석 프레임 조회")
def get_latest_frame():
    if not LAST_FRAME_PATH.exists():
        raise HTTPException(status_code=404, detail="아직 생성된 분석 프레임이 없습니다.")
    return FileResponse(str(LAST_FRAME_PATH), media_type="image/jpeg")


@app.post("/analyze/frame", summary="업로드 이미지 1장 즉시 분석")
async def analyze_frame(file: UploadFile = File(...)):
    if model_status.get("model") != "ok":
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")

    data = await file.read()
    np_arr = bytearray(data)
    frame = cv2.imdecode(
        __import__("numpy").frombuffer(np_arr, dtype=__import__("numpy").uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩에 실패했습니다.")

    result = run_single_frame_analysis(frame)
    return FileResponse(result["frame_path"], media_type="image/jpeg")


@app.post("/video/start", summary="동영상 분석 시작")
def start_video_analysis(req: VideoStartRequest):
    global video_worker_thread

    if model_status.get("model") != "ok":
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")

    video_path = VIDEO_DIR / req.video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"동영상 파일이 없습니다: {video_path}")

    with video_state_lock:
        if video_state["running"]:
            raise HTTPException(status_code=409, detail="이미 다른 동영상 분석이 실행 중입니다.")

    video_stop_event.clear()
    video_worker_thread = threading.Thread(target=video_worker, args=(video_path,), daemon=True)
    video_worker_thread.start()

    return {
        "message": "동영상 분석을 시작했습니다.",
        "video_name": req.video_name,
    }


@app.post("/video/stop", summary="동영상 분석 중지")
def stop_video_analysis():
    with video_state_lock:
        if not video_state["running"]:
            return {"message": "현재 실행 중인 동영상 분석이 없습니다."}

    video_stop_event.set()
    return {"message": "동영상 분석 중지 요청을 보냈습니다."}