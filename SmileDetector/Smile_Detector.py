import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0)

# Show the current frame
while True:
    # Read the current frame from webcam
    successful_frame_read, frame = webcam.read()

    cv2.imshow('Smile Detection', frame)

    # Display
    cv2.waitKey(1)

# Clean up
webcam.release()
cv2.destroyAllWindows()

print('Code works!')