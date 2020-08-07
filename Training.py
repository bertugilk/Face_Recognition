import cv2
import os
from PIL import Image
import numpy as np

recognizer=cv2.face.LBPHFaceRecognizer_create()
CreatePath="Create_Dataset"
ReadyPath="Ready_Dataset"

def createDatasetTrain(CreatePath):
    imagePaths=[os.path.join(CreatePath,f) for f in os.listdir(CreatePath)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNP=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[0])
        faces.append(faceNP)
        print(ID)
        IDs.append(ID)
        cv2.imshow("CreateDatasetTraining",faceNP)
        cv2.waitKey(10)
    return IDs,faces

IDs,faces=createDatasetTrain(CreatePath)

recognizer.train(faces,np.array(IDs))
recognizer.save("Recognizer/trainingData.yml")

cv2.destroyAllWindows()