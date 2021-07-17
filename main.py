import cv2

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

        face = frame[y:y + h, x:x + w]

        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, 1.7, 20)

        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)

    cv2.imshow('smile Detector', frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()
cv2.destroyAllWindows()
