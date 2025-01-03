import cv2
import numpy as np
from pathlib import Path
from detection import detect_mouth_and_create_mask
from loguru import logger

OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)

def process_image(input_image_path: str):
    """
    Process the input image to detect the mouth and create a mask.
    Args:
        input_image_path (str): Path to the input image.
    """
    logger.info("Processing image for face and mouth detection...")

    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Failed to load image. Please check the input path.")

    try:
        # Detect mouth and create a mask
        annotated_image, mouth_mask = detect_mouth_and_create_mask(image)
    except ValueError as e:
        logger.error(f"Error during face or mouth detection: {e}")
        return

    # Save the annotated image
    annotated_image_path = OUTPUT_FOLDER / "annotated_image.jpg"
    cv2.imwrite(str(annotated_image_path), annotated_image)
    logger.info(f"Annotated image saved at: {annotated_image_path}")

    # Save the mouth mask
    mouth_mask_path = OUTPUT_FOLDER / "mouth_mask.jpg"
    cv2.imwrite(str(mouth_mask_path), mouth_mask)
    logger.info(f"Mouth mask saved at: {mouth_mask_path}")

# Replace this with your actual input image path
process_image("images/input.jpg")
