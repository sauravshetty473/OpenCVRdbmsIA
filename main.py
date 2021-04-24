import numpy as np
import cv2
from matplotlib import pyplot as plt



brainMain = cv2.imread('brainMain.png', 0)
blackImage = np.zeros((brainMain.shape[0], brainMain.shape[1], 1), dtype ="uint8")




cv2.imshow('primary image', brainMain)
cv2.imshow('blank image', blackImage)


y, x, _ = plt.hist(brainMain.ravel(), 256, [0, 256])   # histogram

avgIntensity = 0
j=0
maxValue = 0
maxX = 0
for i in y:
    if i > maxValue:
        maxValue = i
        maxX = j

    avgIntensity+=(i*j)
    j = j+1

avgIntensity = avgIntensity/(brainMain.shape[0]*brainMain.shape[1])
plt.show()



edged = cv2.Canny(brainMain, 30, 200)
cv2.imshow('Canny Edges After Contouring', edged, )


contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # getting a single contour over the image
cv2.drawContours(blackImage, contours, contourIdx=-1, color=(255,255,255),thickness=20)   # drawing the contour with thickness 20 over the blank image

cv2.imshow('inverted mask', blackImage)


mask = 255 - blackImage          # inverting the colors for masking
skullRemoved = cv2.bitwise_and(brainMain, mask)    # removing the skull part using the mask obtained above
cv2.imshow('mask', mask)
cv2.imshow('skull removal', skullRemoved)




ret, otsu = cv2.threshold(skullRemoved, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2, simpleThreshold = cv2.threshold(skullRemoved,127,255,cv2.THRESH_BINARY)

cv2.imshow('Otsu', otsu)
cv2.imshow('simple Thresholding', simpleThreshold)



for i in range(0,brainMain.shape[0]):                           # steps required for modified Otsu
    for j in range(0,brainMain.shape[1]):
        if brainMain[i][j]<avgIntensity:
            skullRemoved[i][j] = avgIntensity


cv2.imshow('pre Modified', skullRemoved)
ret, modifiedOtsu = cv2.threshold(skullRemoved, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow('modified otsu', modifiedOtsu)





def getBelow(thres, contours):
    num = 0
    for b in contours:
        if cv2.contourArea(b)<thres:
            num = num+1
    return num


contours, hierarchy = cv2.findContours(modifiedOtsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea,reverse=True)

cv2.drawContours(modifiedOtsu, contours, contourIdx=-1, color=(255,255,255),thickness=-1)

cv2.imshow('last', modifiedOtsu)

mid = []
for i in range(0, int(cv2.contourArea(contours[0]))+1):
    mid.append(i)
mid2 = []
for i in range(0, int(cv2.contourArea(contours[0]))+1):
    mid2.append(getBelow(i,contours))

print(cv2.contourArea(contours[45]))
plt.clf()
plt.plot(mid2, mid)
plt.show()



newContours = []
for i in contours:
    if cv2.contourArea(i) < 100:
        newContours.append(i)


blackImage = np.zeros((brainMain.shape[0], brainMain.shape[1], 1), dtype ="uint8")
cv2.drawContours(blackImage, newContours, contourIdx=-1, color=(255,255,255),thickness=-1)
cv2.imshow('final', blackImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

