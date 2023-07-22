import cv2
import numpy as np


#Preset the image size
widthImage=640
heightImage=480

#video capturing from mac cam
capture = cv2.VideoCapture(0)
capture.set(3,widthImage)
capture.set(4,heightImage)
capture.set(10,150)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThresh = cv2.erode(imgDil,kernel,iterations=1)

    return imgThresh

def getContour(img):
    maxArea = 0
    biggest = np.array([])
    contour,heirarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImage,0],[0,heightImage],[widthImage,heightImage]])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(widthImage,heightImage))
    imgCrop = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCrop = cv2.resize(imgCrop,(widthImage,heightImage))

    return imgCrop



def reorder(Points):
    # Convert Points to a 2D array with shape (4, 2)
    Points = Points.reshape((4, 2))
    newPoints = np.zeros((4,1,2), np.int32)
    # Sum of coordinates along axis 1
    add = Points.sum(1)
    # Find the top-left and bottom-right points
    newPoints[0] = Points[np.argmin(add)]
    newPoints[3] = Points[np.argmax(add)]
    # Difference of coordinates along axis 1
    diff = np.diff(Points, axis=1)
    # Find the top-right and bottom-left points
    newPoints[1] = Points[np.argmin(diff)]
    newPoints[2] = Points[np.argmax(diff)]

    return newPoints

#Display the video footage
while True:
    success,img = capture.read()
    img = cv2.resize(img,(widthImage,heightImage))
    imgContour = img.copy()
    imgThresh = preProcessing(img)
    biggest = getContour(imgThresh)
    if biggest.size != 0 and len(biggest) == 4:
        imgWarp = getWarp(biggest, img)
    else:
        imgWarp = img

    cv2.imshow("Video", imgWarp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





