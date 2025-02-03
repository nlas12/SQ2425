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