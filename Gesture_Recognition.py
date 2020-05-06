import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

chck=0
Open_kernal = np.ones((5,5))
Close_kernal = np.ones((20,20))

mouse = Controller()

app=wx.App(False)
(sx,sy)=wx.GetDisplaySize()
(camx,camy)=(320,240)

cam = cv2.VideoCapture(0)
cam.set(3,camx)
cam.set(4,camy)

oldLoc=np.array([0,0])
curLoc=np.array([0,0])
DF=1.5              #Damping Factor should be greater than 1
#curLoc=oldLoc+(tarLoc+oldLoc)/DF

while True:
    ret, img = cam.read()
    #img = cv2.resize(img,(340,220))
    
    #convert to HSV
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #mask to for Green color
    mask=cv2.inRange(imgHSV,np.array([33,120,40]),np.array([102,255,255]))
    # Morphological mask to remove the noise pixels
    Mor_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Open_kernal)
    Mor_close = cv2.morphologyEx(Mor_open,cv2.MORPH_CLOSE,Close_kernal)
    # To draw contour of Object
    Final_mask = Mor_close
    cont,hr=cv2.findContours(Final_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #ctr = np.array(cont).reshape((-1,1,2)).astype(np.int32)
    #cv2.drawContours(img,cont,-1,(255,0,0),3)
    
    if(len(cont)==2):
        if(chck==1):
            chck=0
            mouse.release(Button.left)
        x1,y1,w1,h1 = cv2.boundingRect(cont[0])
        x2,y2,w2,h2 = cv2.boundingRect(cont[1])
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)
        cx1=int(x1+w1/2)
        cy1=int(y1+h1/2)
        cx2=int(x2+w2/2)
        cy2=int(y2+h2/2)
        cx=int((cx1+cx2)/2)
        cy=int((cy1+cy2)/2)
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)
        cv2.circle(img,(cx,cy),2,(0,0,255),2)

        curLoc=oldLoc+((cx,cy)-oldLoc)/DF
        mouse.position=(int(sx-(curLoc[0]*sx/camx)),int(curLoc[1]*sy/camy))
        while mouse.position!=(int(sx-(curLoc[0]*sx/camx)),int(curLoc[1]*sy/camy)):
            pass
        oldLoc=curLoc
        ArX,ArY,ArW,ArH = cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        
    elif(len(cont)==1):
        x,y,w,h = cv2.boundingRect(cont[0])
        if(chck==0):
            if(abs((w*h-ArW*ArH)*100/(w*h))<20):
                chck=1
                mouse.press(Button.left)
                ArX,ArY,ArW,ArH=(0,0,0,0)
            #else:
             #   chck=1
              #  mouse.press(Button.right)
               # ArX,ArY,ArW,ArH=(0,0,0,0)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cx=int(x+w/2)
            cy=int(y+h/2)
            cv2.circle(img,(cx,cy),int((w+h)/4),(0,0,255),2)

            curLoc=oldLoc+((cx,cy)-oldLoc)/DF
            mouse.position=(int(sx-(curLoc[0]*sx/camx)),int(curLoc[1]*sy/camy))
            while mouse.position!=(int(sx-(curLoc[0]*sx/camx)),int(curLoc[1]*sy/camy)):
                pass
            oldLoc=curLoc

    #Video Streaming
    cv2.imshow("CAMERA",img)   # Original Color Streaming
    
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()
