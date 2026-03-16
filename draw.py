from PIL import Image, ImageDraw, ImageFont
import re

IMAGE_PATH = "test.jpg"
OUTPUT_PATH = "result.jpg"

DETECTION_TEXT = """
[0] class=2 conf=0.82959 box=(1128.3, 64.8, 1533.3, 1002)
[1] class=1 conf=0.823242 box=(1166.1, 327.3, 1533.9, 927.9)
[2] class=1 conf=0.817871 box=(524.25, 346.5, 823.35, 903.9)
[3] class=2 conf=0.8125 box=(526.5, 85.5, 843.9, 1001.7)
[4] class=0 conf=0.805176 box=(1132.2, 87.45, 1368.6, 264.15)
[5] class=2 conf=0.789062 box=(786, 84.9, 1192.8, 1002.3)
[6] class=2 conf=0.779785 box=(106.95, 69.6, 590.25, 979.2)
[7] class=0 conf=0.779297 box=(341.85, 87.15, 567.75, 245.25)
[8] class=0 conf=0.747559 box=(590.4, 93, 792, 247.2)
[9] class=2 conf=0.616211 box=(0, 96, 321.3, 942)
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
