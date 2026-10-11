
# CSC6302 Practical 3 — Object Detection

**Student:** Maluba Mulebi  
**Computer Number:** 2025025124  
**Programme:** MSc Artificial Intelligence  
**Institution:** University of Zambia

## Project Overview

This practical implements traditional and modern computer vision
techniques using Python, OpenCV, YOLOv8n and Gradio.

## Tasks

1. Template matching using normalized cross-correlation.
2. Viola-Jones face detection with a nine-configuration parameter study.
3. YOLOv8n object detection and confidence threshold comparison.
4. Interactive Gradio object detection application.
5. Ethical reflection on face detection in Zambia.

## Requirements

- Python 3.12
- OpenCV
- NumPy
- PyTorch
- Ultralytics
- Gradio
- Matplotlib

## Installation

Clone the repository and navigate into the project folder.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Practical

Task 1:

```bash
python src/task1_template_matching.py
```

Task 2:

```bash
python src/task2_haar_detection.py
```

Task 3:

```bash
python src/task3_yolo_detection.py
```

Launch the Gradio application:

```bash
python app.py
```

Open http://127.0.0.1:7860 in a browser.

The YOLOv8n pretrained weights are downloaded automatically
on first use.

## Project Directories

- `src/` — Python scripts
- `inputs/` — Test images and video
- `outputs/` — Detection results and annotated videos
- `evidence/` — Application screenshots and experimental evidence
- `report/` — Final report and ethics reflection

## Experimental Results

- Template matching: 1.0000 original, 0.2155 changed angle.
- Haar video: 384 frames, 150.15 ms average detection time.
- YOLO video: 384 frames, 146.04 ms average prediction time.
- YOLO bottle detection: confidence approximately 0.72.

The Haar and YOLO timing measurements use different scopes
and should not be interpreted as a strictly equivalent benchmark.

## Ethics

Consent was obtained from the individuals appearing in the
group photograph. The project discusses privacy, fairness,
false positives and responsible face detection.

## AI Assistance

OpenAI ChatGPT was used to assist with coding, debugging,
experimental interpretation and report preparation.
