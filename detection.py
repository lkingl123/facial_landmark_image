import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

mp_face_mesh = mp.solutions.face_mesh

# Updated mouth landmark indices
MOUTH_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84,
    17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

def detect_mouth_and_create_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect mouth landmarks, create a binary mask, and annotate the image.
    Args:
        image (np.ndarray): The input image.
    Returns:
        tuple[np.ndarray, np.ndarray]: Annotated image and the mouth mask.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        # Convert to RGB as Mediapipe requires it
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected")

        # Copy the image for annotations
        annotated_image = image.copy()
        height, width = image.shape[:2]
        mouth_points = []

        # Extract the mouth landmarks
        for face_landmarks in results.multi_face_landmarks:
            for idx in MOUTH_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                mouth_points.append((x, y))

                # Annotate the mouth points with green dots
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)

        # Create the mouth mask
        mouth_mask = np.zeros((height, width), dtype=np.uint8)
        if len(mouth_points) >= 3:  # Ensure at least 3 points for a valid polygon
            cv2.fillPoly(mouth_mask, [np.array(mouth_points, dtype=np.int32)], 255)

        # Log mouth points for debugging
        logger.info(f"Mouth points (filtered): {mouth_points}")
        logger.info(f"Mouth mask created successfully.")

        return annotated_image, mouth_mask
