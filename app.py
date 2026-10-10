
import cv2
import gradio as gr
from ultralytics import YOLO

# Load detection models once when the application starts
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Unable to load Haar face cascade.")

yolo_model = YOLO("yolov8n.pt")


def detect(image, backend, confidence):
    """Detect faces or objects and return an annotated RGB image."""

    if image is None:
        raise gr.Error("Please upload an image first.")

    annotated = image.copy()

    # BACKEND 1: HAAR CASCADE FACE DETECTION
    if backend == "Haar (faces)":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated,
                "Face",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # BACKEND 2: YOLOv8n OBJECT DETECTION
    elif backend == "YOLO (objects)":

        # Gradio supplies RGB; YOLO expects BGR NumPy input
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        results = yolo_model.predict(
            source=image_bgr,
            conf=float(confidence),
            device="cpu",
            verbose=False
        )

        # Convert YOLO's annotated BGR output back to RGB
        annotated_bgr = results[0].plot()
        annotated = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )

    return annotated


# Create the Gradio interface
demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(
            type="numpy",
            image_mode="RGB",
            label="Upload an Image"
        ),
        gr.Radio(
            choices=["Haar (faces)", "YOLO (objects)"],
            value="YOLO (objects)",
            label="Select Detection Method"
        ),
        gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=0.25,
            step=0.05,
            label="YOLO Confidence Threshold"
        )
    ],
    outputs=gr.Image(
        type="numpy",
        image_mode="RGB",
        label="Annotated Detection Result"
    ),
    title="CSC6302 — Computer Vision Object Detection",
    description=(
        "Upload a photograph and choose between traditional "
        "Haar cascade face detection and modern YOLOv8n "
        "object detection. The confidence slider controls "
        "YOLO only; Haar uses fixed detection parameters."
    )
)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
