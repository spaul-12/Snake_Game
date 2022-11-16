import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        #cv2.imshow("food",self.imgFood)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False
    
    def randomFoodLocation(self):
        self.foodPoint=random.randint(100,1000), random.randint(100,600)

    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy
        
        if self.currentLength>self.allowedLength:
            for i,lenght in enumerate(self.lengths):
                self.currentLength-=lenght
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength<self.allowedLength:
                    break
        #check if snake ate the food
        rx,ry=self.foodPoint
        if rx-self.wFood//2<cx<rx+self.wFood//2 and ry-self.hFood//2<cy<ry+self.hFood//2:
            self.randomFoodLocation()
            self.allowedLength+=50
            self.score+=1
            print(self.score)
   
        

        #draw snake
        if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

        
        imgMain=cvzone.overlayPNG(imgMain,self.imgFood,(rx-self.wFood//2, ry- self.hFood//2))
        #collision
        pts = np.array(self.points[:-2],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(imgMain,[pts],False,(0,255,0),3)
        mindist=cv2.pointPolygonTest(pts,(cx,cy),True)
        #print(mindist)

        if -0.2<mindist<0.2:
            print("hit")
            self.gameOver=True
            self.points = []  # all points of the snake
            self.lengths = []  # distance between each point
            self.currentLength = 0  # total length of the snake
            self.allowedLength = 150  # total allowed Length
            self.previousHead = 0, 0  # previous head point
            self.score=0
            self.randomFoodLocation()

        return imgMain

game = SnakeGameClass("Donut.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key &  0xFF==ord('q'):
        break

