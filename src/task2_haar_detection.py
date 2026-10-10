import cv2
import csv
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "inputs/images/group_photo.jpg"
VIDEO_PATH = BASE_DIR / "inputs/videos/street_video.mp4"
OUTPUT_DIR = BASE_DIR / "outputs/task2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Haar cascade
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade.")

# Load group photograph
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError("Group photograph not found.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Experiment with the nine parameter combinations
scale_factors = [1.05, 1.1, 1.3]
min_neighbors_values = [3, 5, 8]

results = []

print("\nTASK 2: HAAR CASCADE PARAMETER STUDY")
print("-------------------------------------")

for scale in scale_factors:
    for neighbors in min_neighbors_values:

        start = time.perf_counter()

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=(30, 30)
        )

        elapsed = (time.perf_counter() - start) * 1000

        annotated = image.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

        filename = (
            f"haar_scale_{scale}_neighbors_{neighbors}.jpg"
        )

        cv2.imwrite(
            str(OUTPUT_DIR / filename),
            annotated
        )

        results.append({
            "scaleFactor": scale,
            "minNeighbors": neighbors,
            "detections": len(faces),
            "processing_ms": round(elapsed, 2)
        })

        print(
            f"Scale={scale}, Neighbors={neighbors}, "
            f"Detections={len(faces)}, "
            f"Time={elapsed:.2f} ms"
        )

# Save parameter study results
csv_path = OUTPUT_DIR / "haar_parameter_study.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "scaleFactor",
            "minNeighbors",
            "detections",
            "processing_ms"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

# Process video frame by frame
video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise FileNotFoundError("Could not open video.")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

output_video = OUTPUT_DIR / "haar_annotated_video.mp4"

writer = cv2.VideoWriter(
    str(output_video),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

frame_count = 0
total_processing_time = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    frame_gray = cv2.equalizeHist(frame_gray)

    start = time.perf_counter()

    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    total_processing_time += time.perf_counter() - start

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, max(y - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    writer.write(frame)
    frame_count += 1

video.release()
writer.release()

average_ms = (
    total_processing_time / frame_count * 1000
    if frame_count else 0
)

print("\nVIDEO RESULTS")
print("----------------")
print(f"Frames processed: {frame_count}")
print(f"Average Haar detection time: {average_ms:.2f} ms")
print(f"Annotated video saved: {output_video}")
print(f"CSV results saved: {csv_path}")