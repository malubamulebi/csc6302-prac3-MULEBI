
import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "inputs" / "images"
VIDEO_PATH = BASE_DIR / "inputs" / "videos" / "street_video.mp4"
OUTPUT_DIR = BASE_DIR / "outputs" / "task3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load pretrained YOLOv8n model on CPU
model = YOLO("yolov8n.pt")
thresholds = [0.10, 0.25, 0.50, 0.75]

images = {
    "group_photo": IMAGE_DIR / "group_photo.jpg",
    "object_photo": IMAGE_DIR / "object_photo.jpg"
}

results_table = []

print("\nTASK 3: YOLO CONFIDENCE THRESHOLD STUDY")
print("----------------------------------------")

# Run YOLO on both photographs
for image_name, image_path in images.items():
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Missing image: {image_path}")

    for confidence in thresholds:
        start = time.perf_counter()

        # For the group photo, detect the person class only
        classes = [0] if image_name == "group_photo" else None

        results = model.predict(
            source=image,
            conf=confidence,
            classes=classes,
            device="cpu",
            verbose=False
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        result = results[0]
        detections = len(result.boxes)

        annotated = result.plot()
        filename = f"{image_name}_conf_{confidence:.2f}.jpg"

        cv2.imwrite(str(OUTPUT_DIR / filename), annotated)

        results_table.append({
            "image": image_name,
            "confidence": confidence,
            "detections": detections,
            "processing_ms": round(elapsed_ms, 2)
        })

        print(
            f"{image_name} | confidence={confidence:.2f} | "
            f"detections={detections} | time={elapsed_ms:.2f} ms"
        )

# Save confidence threshold comparison
with open(OUTPUT_DIR / "yolo_confidence_study.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image", "confidence", "detections", "processing_ms"
        ]
    )
    writer.writeheader()
    writer.writerows(results_table)

# Process video and measure inference time
video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "yolo_annotated_video.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Unable to write YOLO annotated video.")

frame_count = 0
total_time = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    start = time.perf_counter()

    prediction = model.predict(
        source=frame,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    total_time += time.perf_counter() - start

    annotated = prediction.plot()
    writer.write(annotated)

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} video frames")

video.release()
writer.release()

average_ms = total_time / frame_count * 1000 if frame_count else 0

print("\nVIDEO RESULTS")
print("----------------")
print(f"Frames processed: {frame_count}")
print(f"Average YOLO inference time: {average_ms:.2f} ms")
print(f"Annotated video saved to: {OUTPUT_DIR}")
print("Confidence study saved as yolo_confidence_study.csv")
