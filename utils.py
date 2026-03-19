import cv2
import numpy as np
from typing import Tuple


def process_damage_image(image_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Improved damage detection using OpenCV
    """

    original = image_array.copy()

    # -----------------------------
    # STEP 1: Crop ROI (ignore background)
    # -----------------------------
    h, w = image_array.shape[:2]
    roi = image_array[int(h * 0.2):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    processed_image = original.copy()

    # -----------------------------
    # STEP 2: Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # -----------------------------
    # STEP 3: Blur to reduce noise
    # -----------------------------
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # -----------------------------
    # STEP 4: Threshold (focus on strong damage contrast)
    # -----------------------------
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # -----------------------------
    # STEP 5: Edge detection
    # -----------------------------
    edges = cv2.Canny(thresh, 50, 150)

    # -----------------------------
    # STEP 6: Dilate to strengthen edges
    # -----------------------------
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # -----------------------------
    # STEP 7: Find contours
    # -----------------------------
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = roi.shape[0] * roi.shape[1]
    damage_area = 0

    # Offset for drawing on original image
    y_offset = int(h * 0.2)
    x_offset = int(w * 0.1)

    # -----------------------------
    # STEP 8: Filter only large contours
    # -----------------------------
    for contour in contours:
        area = cv2.contourArea(contour)

        # 🔥 Increased threshold to remove noise
        if area > 2000:
            damage_area += area

            # Shift contour back to original image position
            contour_shifted = contour + [x_offset, y_offset]

            cv2.drawContours(processed_image, [contour_shifted], -1, (255, 0, 0), 3)

            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.rectangle(
                processed_image,
                (x + x_offset, y + y_offset),
                (x + w_box + x_offset, y + h_box + y_offset),
                (0, 255, 0),
                2
            )

    # -----------------------------
    # STEP 9: Calculate damage %
    # -----------------------------
    damage_percentage = (damage_area / total_area) * 100

    # 🔥 Cap unrealistic values
    damage_percentage = min(damage_percentage * 1.5, 85)

    return processed_image, damage_percentage


def classify_damage_severity(damage_percent: float) -> str:
    if damage_percent <= 20:
        return "Minor"
    elif damage_percent <= 50:
        return "Moderate"
    else:
        return "Severe"


def get_severity_color(severity: str) -> str:
    color_map = {
        "Minor": "green",
        "Moderate": "orange",
        "Severe": "red"
    }
    return color_map.get(severity, "blue")