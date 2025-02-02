import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

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

        # Convert image from BGR (OpenCV) to RGB (DeepFace expects RGB)
        rgb_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        try:
            # Perform emotion recognition
            predictions = DeepFace.analyze(rgb_image, actions=['emotion'], enforce_detection=False)

            # Extract highest confidence emotion
            dominant_emotion = predictions[0]['dominant_emotion']
            confidence = predictions[0]['emotion'][dominant_emotion]

            text = f"{dominant_emotion.upper()} ({confidence:.2f}%)"

            # Convert to PIL image for editing
            pil_image = Image.fromarray(rgb_image)
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()

            # Draw text in the top-left corner
            draw.text((10, 10), text, font=font, fill=(0, 255, 0))
            pil_image.show()

        except Exception as e:
            print(f"Error during emotion recognition: {e}")

    elif key == ord('q'):
        print("Stopping...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()