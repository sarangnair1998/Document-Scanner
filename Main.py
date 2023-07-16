import cv2
import numpy as np

image = cv2.imread("instructions.jpeg")
imgcopy = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
height,width = imgcopy.shape[:2]

_,threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(largest_contour,True)
epsilon = 0.02*peri
approx = cv2.approxPolyDP(largest_contour,epsilon,True)

corner_points = approx.reshape(4, 2)
for corner in corner_points:
    cv2.circle(imgcopy,tuple(corner),20,(255,0,0),-1)

cv2.drawContours(imgcopy, [largest_contour], -1, (0, 255, 0), 2)

destination_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
corner_points = np.float32(corner_points[::-1])
destination_points = np.array(destination_points, dtype=np.float32)

perspective_matrix = cv2.getPerspectiveTransform(corner_points, destination_points)
warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))

cv2.drawContours(imgcopy, [largest_contour], -1, (0, 255, 0), 2)
cv2.imshow("Bird's Eye view",warped_image)
cv2.waitKey(0)





# while True:
#     h_min = cv2.getTrackbarPos("Hue Min", "ColorPicker")
#     h_max = cv2.getTrackbarPos("Hue Max", "ColorPicker")
#     s_min = cv2.getTrackbarPos("Sat Min", "ColorPicker")
#     s_max = cv2.getTrackbarPos("Sat Max", "ColorPicker")
#     v_min = cv2.getTrackbarPos("Val Min", "ColorPicker")
#     v_max = cv2.getTrackbarPos("Val Max", "ColorPicker")
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask = cv2.inRange(imgHSV,lower,upper)
#
#
#     cv2.imshow("image",imgHSV)
#     cv2.imshow("mask", mask)
#
#     cv2.waitKey(1)
#
#
#
# # x,y,w,h = cv2.boundingRect(max_contour)
# #
# # cv2.rectangle(imgcopy,(x,y),(x+w,y+h),(0,255,0),1)
# #
# #
# # contourimage = np.zeros_like(imgcopy)
# #
# # #cv2.drawContours(imgcopy,contours,-1,(0,255,0),2)






