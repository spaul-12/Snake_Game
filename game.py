import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)
clr_r=255
clr_g=0
clr_b=0

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point
        self.clr_r=255
        self.clr_b=0
        self.clr_g=0
        self.imgFood =cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        #cv2.imshow("food",self.imgFood)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead,clr_r,clr_b,clr_g):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead
            
            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy
            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake ate the Food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(self.score)
                clr_r=random.randint(20,255)
                clr_b=random.randint(20,255)
                clr_g=random.randint(20,255)

            # Draw Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (clr_g,clr_b,clr_r), 20)
                cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                        (rx - self.wFood // 2, ry - self.hFood // 2))

            cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],
                               scale=3, thickness=3, offset=10)

            # Check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1<= minDist <=1:
                print("Hit")
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()

        return (imgMain,clr_r,clr_b,clr_g)


game = SnakeGameClass("tux(2.1).png")

while True:
    success, img = cap.read()
    width=int(img.shape[1]*2)
    height=int(img.shape[0]*2)
    dimension=(width,height)
    img=cv2.resize(img,dimension,interpolation=cv2.INTER_CUBIC)
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        #print("1")
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        #print(clr_r)
        img,clr_r,clr_b,clr_g = game.update(img, pointIndex,clr_r,clr_b,clr_g)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
    elif key & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#key == ord('r')
