import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "inputs" / "images"
OUTPUT_DIR = BASE_DIR / "outputs" / "task1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the original and different-angle photographs
original = cv2.imread(str(INPUT_DIR / "object_photo.jpg"))
angled = cv2.imread(str(INPUT_DIR / "object_photo_angle.jpg"))

if original is None or angled is None:
    raise FileNotFoundError(
        "Could not load the bottle photographs. Check the filenames."
    )

# Convert both images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
angled_gray = cv2.cvtColor(angled, cv2.COLOR_BGR2GRAY)

# Crop a distinctive region containing the bottle label
# Coordinates are expressed as proportions of image dimensions.
height, width = original_gray.shape

x1 = int(width * 0.30)
x2 = int(width * 0.72)
y1 = int(height * 0.52)
y2 = int(height * 0.78)

template = original_gray[y1:y2, x1:x2]

if template.size == 0:
    raise ValueError("Template crop is empty.")

cv2.imwrite(str(OUTPUT_DIR / "bottle_template.jpg"), template)

# Resize the angled image only if it is smaller than the template
th, tw = template.shape

if angled_gray.shape[0] < th or angled_gray.shape[1] < tw:
    scale = max(
        th / angled_gray.shape[0],
        tw / angled_gray.shape[1]
    )
    angled_gray = cv2.resize(
        angled_gray,
        None,
        fx=scale,
        fy=scale
    )
    angled = cv2.resize(
        angled,
        (angled_gray.shape[1], angled_gray.shape[0])
    )

# Apply normalized cross-correlation
method = cv2.TM_CCOEFF_NORMED

original_scores = cv2.matchTemplate(
    original_gray, template, method
)

angled_scores = cv2.matchTemplate(
    angled_gray, template, method
)

# Find the strongest correlation in each image
_, original_max, _, original_loc = cv2.minMaxLoc(original_scores)
_, angled_max, _, angled_loc = cv2.minMaxLoc(angled_scores)

# Draw bounding boxes around the strongest matches
original_result = original.copy()
angled_result = angled.copy()

for image, location in [
    (original_result, original_loc),
    (angled_result, angled_loc)
]:
    cv2.rectangle(
        image,
        location,
        (location[0] + tw, location[1] + th),
        (0, 255, 0),
        4
    )

# Save annotated detection results
cv2.imwrite(
    str(OUTPUT_DIR / "original_detection.jpg"),
    original_result
)

cv2.imwrite(
    str(OUTPUT_DIR / "angled_detection.jpg"),
    angled_result
)

# Save both correlation score maps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(
    original_scores, cmap="viridis", vmin=-1, vmax=1
)
axes[0].set_title(f"Original image: {original_max:.3f}")
axes[0].set_xlabel("X position")
axes[0].set_ylabel("Y position")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(
    angled_scores, cmap="viridis", vmin=-1, vmax=1
)
axes[1].set_title(f"Changed angle: {angled_max:.3f}")
axes[1].set_xlabel("X position")
axes[1].set_ylabel("Y position")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "correlation_score_maps.png",
    dpi=200
)
plt.close()

# Print experimental results
print("\nTASK 1: TEMPLATE MATCHING RESULTS")
print("----------------------------------")
print(f"Original image correlation: {original_max:.4f}")
print(f"Changed-angle correlation:  {angled_max:.4f}")
print(f"Difference: {original_max - angled_max:.4f}")
print(f"\nResults saved to: {OUTPUT_DIR}")