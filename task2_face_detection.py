# ================================================================
#   TASK 2 — FACE DETECTION APP
#   Kodbud AI Internship
#   Just run: python task2_face_detection.py
# ================================================================

import cv2

print("=" * 50)
print("   FACE DETECTION APP  |  Kodbud AI Internship")
print("=" * 50)

# ----------------------------------------------------------------
# STEP 1 : LOAD HAAR CASCADE
# ----------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("\n  [OK] Haar Cascade loaded!")
print("  [OK] OpenCV ready!")

# ----------------------------------------------------------------
# STEP 2 : START WEBCAM
# ----------------------------------------------------------------
print("\n  Starting webcam...")
print("  Press  Q  to quit")
print("  Press  S  to save screenshot")
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("  [ERROR] Webcam not found!")
    print("  Make sure your webcam is connected.")
    exit()

print("  [OK] Webcam opened! Show your face :)")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("  [ERROR] Cannot read from webcam.")
        break

    frame_count += 1

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangle around each face
    for (x, y, w, h) in faces:

        # Blue rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 3)

        # Label above rectangle
        cv2.putText(frame, 'Face', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

        # Detect eyes inside face
        roi_gray  = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Top bar showing face count
    cv2.rectangle(frame, (0, 0), (320, 45), (0, 0, 0), -1)
    cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

    # Show the frame
    cv2.imshow('Face Detection - Kodbud AI Internship', frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n  Closing webcam...")
        break

    elif key == ord('s') or key == ord('S'):
        filename = f'face_screenshot_{frame_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f"  [SAVED] Screenshot saved: {filename}")

# ----------------------------------------------------------------
# STEP 3 : CLEANUP
# ----------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("  Task 2 Complete! Face Detection working!  [OK]")
print("=" * 50)
