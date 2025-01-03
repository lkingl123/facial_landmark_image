import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

mp_face_mesh = mp.solutions.face_mesh

# Define teeth-specific landmarks
TEETH_LANDMARKS = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 325, 315, 14, 87, 178, 88, 95
]


def detect_teeth_and_create_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect teeth landmarks, create a binary mask, and annotate the image.
    Args:
        image (np.ndarray): The input image.
    Returns:
        tuple[np.ndarray, np.ndarray]: Annotated image and the refined teeth mask.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected")

        annotated_image = image.copy()
        height, width = image.shape[:2]
        teeth_points = []

        # Extract teeth landmarks only
        for face_landmarks in results.multi_face_landmarks:
            for idx in TEETH_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Filter out points that are too far from the teeth region
                if 0.2 * width < x < 0.8 * width and 0.4 * height < y < 0.7 * height:
                    teeth_points.append((x, y))
                    cv2.circle(annotated_image, (x, y), 2, (0, 255, 255), -1)  # Annotate with yellow dots

        # Debug: Log filtered teeth points
        logger.info(f"Teeth points (filtered): {teeth_points}")

        # Ensure there are enough points to create a mask
        if len(teeth_points) < 6:  # Minimum points to create a valid teeth region
            raise ValueError("Insufficient teeth landmarks detected")

        # Create the teeth mask
        teeth_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(teeth_mask, [np.array(teeth_points, dtype=np.int32)], 255)

        # Refine the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined_teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel)

        # Save debug images
        debug_path = "output/teeth_mask_debug.jpg"
        cv2.imwrite(debug_path, refined_teeth_mask)
        logger.info(f"Teeth mask saved for debugging: {debug_path}")

        logger.info("Teeth mask created and refined successfully.")
        return annotated_image, refined_teeth_mask
