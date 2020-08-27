import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0)

# Show the current frame
while True:
    # Read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read()

    # If there's an error, abort
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    print(faces)

    # Show the current frame
    cv2.imshow('Smile Detection', frame_grayscale)

    # Display
    cv2.waitKey(1)

# Clean up
webcam.release()
cv2.destroyAllWindows()

# Code ran without errors
print('Code works!')