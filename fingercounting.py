import cv2
import time
import os
import handtrackingmodule as htm



wCam,hCam=640,490
cap=cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)
folderpath="fingerimages"
myList=os.listdir(folderpath)

print(myList)


overlayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderpath}/{imPath}')
    #print(f'{folderpath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime=0

detector=htm.handDetector(detectionCon=0.75)

tipsIds=[4,8,12,16,20]


while True:
    success,img=cap.read()

    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)
    #print(lmlist)

    if(len(lmlist)!=0):
        fingers= []

        #for thumb
        if (lmlist[tipsIds[0]][1]  > lmlist[tipsIds[0] - 1][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        #for fingers


        for id in range(1,5):

            if(lmlist[tipsIds[id]][2]<lmlist[tipsIds[id]-2][2]):
                fingers.append(1)
            else:
                fingers.append(0)
                #print("index finger open")

        #print(fingers)
        totalFingers=fingers.count(1)

        print(totalFingers)


        h,w,c=overlayList[totalFingers-1].shape
        img[0:h,0:w]=overlayList[totalFingers-1]

        cv2.rectangle(img, (20,225),(178,425), (0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0),25)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, f'FPS : {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,
                3,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
