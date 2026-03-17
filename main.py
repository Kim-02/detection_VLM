import threading
import queue
import time
import cv2
from PIL import Image

import yolo_detection
import smolVLrun

from ultralytics import YOLO

event_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

vlm_busy = False
vlm_busy_lock = threading.Lock()
last_vlm_trigger_time = 0.0
VLM_TRIGGER_COOLDOWN = 5.0

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

        # 기본 색상
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
            2
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
        2
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

def resize_to_640(frame):
    return cv2.resize(frame, (640, 640))

def is_vlm_busy():
    with vlm_busy_lock:
        return vlm_busy


def set_vlm_busy(value: bool):
    global vlm_busy
    with vlm_busy_lock:
        vlm_busy = value


def build_vlm_prompt():
    return """
Fire is present.
Reply in English only.

Format:
Scene: <one short sentence>

Rules:
- Describe what is happening in the scene.
- Mention visible surroundings briefly.
- Keep it short.
- Do not write anything else.
""".strip()

def vlm_worker(runner):
    while not stop_event.is_set():
        try:
            item = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        set_vlm_busy(True)

        try:
            frame = item["frame"]
            analysis = item["analysis"]
            timestamp = item["timestamp"]

            # OpenCV BGR -> PIL RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            prompt = build_vlm_prompt()

            result_text = runner.infer(
                image_input=pil_image,
                user_text=prompt,
                max_new_tokens=64
            )

            if result_text:
                print("[VLM 결과]")
                print(result_text)
            else:
                print("[VLM 결과] 유효한 문장을 생성하지 못했습니다.")

        except Exception as e:
            print(f"[VLM WORKER][오류] {e}")

        finally:
            set_vlm_busy(False)
            event_queue.task_done()


def main():
    model = YOLO("best.pt")
    runner = smolVLrun.SmolVLMRunner()

    cap = cv2.VideoCapture("people_fire.mp4")
    if not cap.isOpened():
        print("영상 파일을 열 수 없습니다: non_people_fire.mp4")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_interval = 1.0 / fps

    print(f"[INFO] video fps: {fps:.2f}, frame interval: {frame_interval:.4f}s")

    worker = threading.Thread(target=vlm_worker, args=(runner,), daemon=True)
    worker.start()

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        resize_frame = resize_to_640(frame)
        detections = yolo_detection.detect_positions_with_class_on_frame(model, resize_frame)
        analysis = analyze_detected_classes(detections)

        display_frame = draw_detections(resize_frame, detections)
        display_frame = draw_status(display_frame, analysis)
        cv2.imshow("frame", display_frame)

        global last_vlm_trigger_time

        if analysis["has_fire"]:
            now = time.time()
            if now - last_vlm_trigger_time >= VLM_TRIGGER_COOLDOWN:
                if not is_vlm_busy() and event_queue.empty():
                    event_queue.put({
                        "frame": resize_frame.copy(),
                        "analysis": analysis,
                        "timestamp": now,
                    })
                    last_vlm_trigger_time = now
                    print("[MAIN] VLM 이벤트 전달")

        elapsed = time.time() - loop_start
        remaining = frame_interval - elapsed
        delay_ms = max(1, int(remaining * 1000))

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()