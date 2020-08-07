import cv2

camera=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier(r'C:\Users\bertug\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

sayac=0

id=int(input("Picture ID: "))

while True:
    _,rect=camera.read()
    gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY)
    faces=cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sayac+=1
        cv2.rectangle(rect,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imwrite("Create_Dataset/"+str(id)+"."+str(sayac)+".jpg",gray[y:y+h,x:x+w])
        cv2.waitKey(100)
    cv2.imshow("Face",rect)
    if sayac>20:
        break

camera.release()

cv2.destroyAllWindows()