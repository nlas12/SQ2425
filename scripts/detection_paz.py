"""
Emotion Recognition from Webcam Feed

Author: Niklas Bockholt
Date: 2025-02-09
License: MIT

Description:
This script captures real-time video from a webcam, allowing the user to take snapshots and analyze facial emotions using the DeepFace library. The detected emotion is displayed on the image.

Features:
- Captures video from the webcam.
- Press 'c' to capture an image and analyze emotions.
- Uses DeepFace to recognize the dominant emotion.
- Displays the recognized emotion on the captured image.
- Press 'q' to exit the application.

Dependencies:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- paz (oarriaga/PAZ)
- PIL (`Pillow`)

Usage:
1. Clone the PAZ repository and place the `paz` directory in your project path
2. Run the script.
3. When the webcam feed appears, press 'c' to capture an image and analyze emotions.
4. The detected emotion will be displayed on the image.
5. Press 'q' to exit.

Error Handling:
- If the webcam is not found, the script will exit with an error message.
- If emotion detection fails, an error message is displayed instead.

"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paz.pipelines import MiniXceptionFER

# Initialize emotion classifier
classifier = MiniXceptionFER()

print("Starting Script...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam Found.")


print("Running...")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam - Press 'c' to Capture - Press 'q' to exit", frame)

    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, capture image and classify
    if key == ord('c'):
        captured_image = frame

        # Convert image from BGR (OpenCV) to RGB (PAZ expects RGB)
        rgb_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        # Predict and show image
        predictions = classifier(rgb_image)
        emotion = predictions["class_name"]
        confidence = np.max(predictions["scores"])  # Highest score as certainty

        text = f"{emotion.upper()} ({confidence:.2%})"

        # Convert to PIL image for editing
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()

        draw.text((10, 10), text, font=font, fill=(0, 255, 0))
        pil_image.show()


    elif key == ord('q'):
        print("Stopping...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()