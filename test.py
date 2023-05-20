# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

# Read Image, convert to grayscale to have less features.
img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
#plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

# 
# no need to do edged.copy()
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(imutils.find_function("grab"))
contours = imutils.grab_contours(keypoints) # converting keypoints to iterable
print(f"{len(contours)=}")
# Take the 10 biggest contours, sorted from large to small.
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#rec = cv2.polylines(img, contours[0], True, (0,255,0),3)
#for i in range(10):
#    rec = cv2.polylines(img, contours[i], True, (0,255,0),3)


location = None
#i = 0
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True) # find corners of contours
    # FOR UNDERSTANDING:
    #if i == 0: # show corners of biggest contour
    #    print(approx)
    #    rec = cv2.polylines(img, approx, True, (0,255,0),10)
    #    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(approx) == 4: # if 4 corners, we found it
        location = approx
        break
    #i += 1

#print(location)


mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

reader = easyocr.Reader(['en'], recog_network='custom_example')
result = reader.readtext(cropped_image)

# "result" contains all recognised text, even if it is in different places.
# bounds, text, confidence = result[0]
text = result[0][-2]


font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
