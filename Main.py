import cv2
camera=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier(r'C:\Users\bertug\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("Recognizer\\trainingData.yml")

id=0

while True:
    _, rect = camera.read()
    gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY)
    faces=cascade.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])

        if (id == 1):
            id = "ALEX"
        else:
            id="Anonymous"

        cv2.putText(rect, str(id), (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    cv2.imshow("Face", rect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()