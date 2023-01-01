import click
import cv2
import dlib
import numpy as np
import os

from pathlib import Path
from PIL import Image, ImageFile


class Rotator:
    IMAGES_DIRECTORY = "/images"

    def __init__(self, overwrite_files: bool=False):
        self.detector = dlib.get_frontal_face_detector()
        self.overwrite_files =overwrite_files

    def analyze_images(self,file_path):  
        image = self.open_image(file_path)
        rotation = self.analyze_image(image, file_path)
        if rotation:
          print("Image rotated")


    def analyze_image(self, image: ImageFile, filepath: str) -> int:
        """Cycles through 4 image rotations of 90 degrees.
           Saves the image at the current rotation if faces are detected.
        """

        for cycle in range(0, 4):
            if cycle > 0:
                # Rotate the image an additional 90 degrees for each non-zero cycle.
                image = image.rotate(90, expand=True)

            image_copy = np.asarray(image)
            image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

            faces = self.detector(image_gray, 0)
            if len(faces) == 0:
                continue

            # Save the image only if it has been rotated.
            if cycle > 0:
                self.save_image(image, filepath)
                return cycle * 90

        return 0

    def open_image(self, filepath: str) -> ImageFile:
        """Intentionally opens an image file using Pillow.
           If opened with OpenCV, the saved image is a much larger file size than the original
           (regardless of whether saved via OpenCV or Pillow).
        """

        return Image.open(filepath)

    def save_image(self, image: ImageFile, filepath: str) -> bool:  
        """Saves the rotated image using Pillow."""

        if not self.overwrite_files:
            filepath = filepath.replace(".", "-rotated.", 1)

        try:
            print("Saving")
            image.save(filepath)
            print("Saved")
            return True
        except:
            return False

