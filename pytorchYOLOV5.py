import torch

import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import csv



# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video stream
cap = cv.VideoCapture("vid_Trim.mp4")  # 0 indicates that we want to use the default camera
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv.VideoWriter_fourcc(*'MJPG')
videoOUT = cv.VideoWriter('YOLOV5/output.avi',fourcc, 30, (width, height), isColor=True)

fpsList = []
confList = []
elapsedTime = 0

while True:
    # Grab the current frame
    ret, frame = cap.read()

    # Check if the frame was successfully grabbed
    if not ret:
        break

    fps = 0
    tempConfs = [0]

    t0 = time.time()
    output = model(frame)
    t = time.time() - t0
    elapsedTime+=t

    fps = 1000/t
    # fps = t
    df = output.pandas().xyxy[0]
    for index, row in df.iterrows():
        xmax = int(row['xmax'])
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        ymax = int(row['ymax'])
        print(xmin, ymin, xmax, ymax, int(row['confidence']*100), row['name'])
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
        cv.putText(frame, row['name']+":"+str(int(row['confidence']*100))+"%", (xmin, ymin - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tempConfs.append(int(row['confidence']*100))

    fpsList.append(fps)
    confList.append(tempConfs[-1])

    cv.putText(frame, "fps: "+str(int(fps)), (25, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    videoOUT.write(frame)

cap.release()
videoOUT.release()

plt.plot(range(0,len(confList)), confList)
plt.xlabel('Frame')
plt.ylabel('Confidence')
plt.title('Frame to Confidence - YOLOV5')
plt.savefig('YOLOV5/FCGraph.png')
plt.clf()
plt.plot(range(0,len(fpsList)), fpsList)
plt.xlabel('Frame')
plt.ylabel('FPS')
plt.title('Frame to FPS - YOLOV5 - Total Time: '+ str(elapsedTime) + 'ms')
plt.savefig('YOLOV5/FFGraph.png')

with open('YOLOV5/output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(0,len(confList)):
        writer.writerow([i, confList[i], fpsList[i]])
