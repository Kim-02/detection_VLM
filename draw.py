from PIL import Image, ImageDraw, ImageFont
import re

IMAGE_PATH = "test.jpg"
OUTPUT_PATH = "result.jpg"

DETECTION_TEXT = """
Detections: 10
[0] class=1 conf=0.854004 box=(1000.5, 419.1, 1171.5, 660.9)
[1] class=1 conf=0.844238 box=(1210.2, 417.6, 1353, 640.8)
[2] class=2 conf=0.759277 box=(1178.7, 282, 1372.5, 922.8)
[3] class=2 conf=0.739258 box=(963, 320.7, 1200.6, 944.1)
[4] class=0 conf=0.68457 box=(1211.7, 313.2, 1317.9, 399)
[5] class=2 conf=0.559082 box=(311.25, 327.15, 429.15, 602.85)
[6] class=1 conf=0.496094 box=(324.6, 381.6, 407.4, 472.8)
[7] class=0 conf=0.455078 box=(341.85, 330.6, 396.15, 379.2)
[8] class=1 conf=0.331299 box=(915.9, 382.5, 996.9, 464.7)
[9] class=2 conf=0.253906 box=(902.1, 320.55, 1022.7, 550.65)
"""

CLASS_NAMES = {
    0: "helmet",
    1: "vest",
    2: "person",
    3: "danger_vehicle",
}

CLASS_COLORS = {
    0: (255, 0, 0),      # helmet - red
    1: (0, 255, 0),      # vest - green
    2: (0, 120, 255),    # person - blue
    3: (255, 140, 0),    # danger_vehicle - orange
}

def parse_detections(text: str):
    pattern = re.compile(
        r"\[\d+\]\s+class=(\d+)\s+conf=([0-9.]+)\s+box=\(([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)\)"
    )

    detections = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = pattern.search(line)
        if not m:
            continue

        detections.append({
            "class_id": int(m.group(1)),
            "conf": float(m.group(2)),
            "x1": float(m.group(3)),
            "y1": float(m.group(4)),
            "x2": float(m.group(5)),
            "y2": float(m.group(6)),
        })
    return detections

def load_font(size=22):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def clamp(v, low, high):
    return max(low, min(v, high))

def main():
    detections = parse_detections(DETECTION_TEXT)
    image = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(22)

    img_w, img_h = image.size

    for det in detections:
        class_id = det["class_id"]
        conf = det["conf"]

        x1 = int(round(clamp(det["x1"], 0, img_w - 1)))
        y1 = int(round(clamp(det["y1"], 0, img_h - 1)))
        x2 = int(round(clamp(det["x2"], 0, img_w - 1)))
        y2 = int(round(clamp(det["y2"], 0, img_h - 1)))

        color = CLASS_COLORS.get(class_id, (255, 255, 0))
        class_name = CLASS_NAMES.get(class_id, f"class{class_id}")
        label = f"{class_name} {conf:.2f}"

        # box
        thickness = 4
        for t in range(thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=color
            )

        # label background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = x1
        label_y = y1 - text_h - 8
        if label_y < 0:
            label_y = y1 + 4

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 10, label_y + text_h + 6],
            fill=color
        )
        draw.text(
            (label_x + 5, label_y + 3),
            label,
            fill=(255, 255, 255),
            font=font
        )

    image.save(OUTPUT_PATH, quality=95)
    print(f"saved: {OUTPUT_PATH}")
    print(f"detections: {len(detections)}")

if __name__ == "__main__":
    main()
