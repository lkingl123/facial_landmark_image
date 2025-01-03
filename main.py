import cv2
import numpy as np
from pathlib import Path
from detection import detect_teeth_and_create_mask
from loguru import logger

OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)

def process_image(input_image_path: str):
    """
    Process the input image to detect the teeth and create a mask.
    Args:
        input_image_path (str): Path to the input image.
    """
    logger.info("Processing image for teeth detection...")

    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Failed to load image. Please check the input path.")

    try:
        # Detect teeth and create a mask
        debug_image, teeth_mask = detect_teeth_and_create_mask(image)
    except ValueError as e:
        logger.error(f"Error during teeth detection: {e}")
        return

    # Save the debug image
    debug_image_path = OUTPUT_FOLDER / "debug_image.jpg"
    cv2.imwrite(str(debug_image_path), debug_image)
    logger.info(f"Debug image saved at: {debug_image_path}")

    # Save the teeth mask
    teeth_mask_path = OUTPUT_FOLDER / "teeth_mask.jpg"
    cv2.imwrite(str(teeth_mask_path), teeth_mask)
    logger.info(f"Teeth mask saved at: {teeth_mask_path}")


# Replace this with your actual input image path
process_image("images/input.jpg")
