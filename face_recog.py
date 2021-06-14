Face Recognition – Unlock Your Computer With Your Face!
Step 1 - Create Training Data
import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
Step 2 - Train Model
cap.release()
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

harsh_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
harsh_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")
Model trained sucessefully
Step 3 - Run Our Facial Recognition
import os
import subprocess as sp

def ip():
    file = open("ip.txt","r")
    all_lines = file.readlines()
    return all_lines[9].split()[2]


def gkscluster():
    print("creating gks cluster............")
    sp.getoutput("terraform   -chdir=C:\\Users\\harsh\\Desktop\\test  apply   -auto-approve")
    print("cluster created.........")
    print("Getting credentials")
    os.system("gcloud container clusters get-credentials my-gke-cluster --region us-central1 --project lw-summer")
    print("Creating Deployment...")
    os.system("kubectl create deploy myweb --image=vimal13/apache-webserver-php:latest")
    print("Deployment Created.")
    print("Exposing Service...")
    os.system("kubectl expose deploy myweb --port=80 --type=LoadBalancer")
    print("service Exposed.")
    os.system("kubectl describe services myweb > ip.txt")
    print("Getting IP...")
    loadBalancer = ip()
    print(loadBalancer)
    Endpoint = "chrome " + loadBalancer
    print("Let's See How it Looks like....")
    os.system(Endpoint)
import cv2
import numpy as np
import os
import pywhatkit


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = harsh_model.predict(face)
        # harry_model.predict(face)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 90:
            cv2.putText(image, "Hello Harsh ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            # os.system("chrome https://www.google.com/search?q=harsh+goenka")
            #os.system("wmplayer   c:\lw.mp3")
            gkscluster()
            break
         
        else:
            
            cv2.putText(image, "I dont know, how r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        

cv2.destroyAllWindows() 
cap.release()

 
 
