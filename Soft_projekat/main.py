import numpy as np
import cv2
import Person
import time

#Petlja za obradu svih klipova

for p in range(1, 11):
    cap = cv2.VideoCapture('video' + str(p) + '.mp4')

#Brojaci kretanja na gore, dole, i ukupno
    broj_gore   = 0
    broj_dole = 0
    broj_ukupno = 0


    w = cap.get(3)
    h = cap.get(4)
    frameArea = h*w
    areaTH = frameArea/30000 #Threshold osobe
    print 'Area Threshold', areaTH

    #Linije prelaza na gore i dole
    linija_gore = float(2*(h/4))
    linija_dole   = float(2*(h/4))

    gornja_granica =   int(1*(h/4))
    donja_granica = int(3*(h/4))

    linija_dole_color = (255,0,0)
    linija_gore_color = (0,0,255)
    pt1 =  [0, linija_dole];
    pt2 =  [w, linija_dole];
    pts_L1 = np.array([pt1,pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1,1,2))
    pt3 =  [0, linija_gore];
    pt4 =  [w, linija_gore];
    pts_L2 = np.array([pt3,pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1,1,2))

    pt5 =  [0, gornja_granica];
    pt6 =  [w, gornja_granica];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 =  [0, donja_granica];
    pt8 =  [w, donja_granica];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))

    #Subtrakcija pozadine
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

    #Elementi filtera
    kernelOp = np.ones((3,3),np.uint8)
    kernelOp2 = np.ones((5,5),np.uint8)
    kernelCl = np.ones((11,11),np.uint8)

    #promenljive
    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 5
    pid = 1

    while(cap.isOpened()):
        #Citaj frejm
        ret, frame = cap.read()

        for i in persons:
            i.age_one() 
      
            
        #Primeni oduzimanje pozadine
        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg.apply(frame)

        #Ciscenje buke
        try:
            ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
            ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
            #Otvaranje (erode->dilate)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
            #Zatvaranje (dilate -> erode)
            mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        except:
            print('EOF')
            print 'UP:',broj_gore
            print 'DOWN:',broj_dole
            break
        
        #Konture       
        contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours0:
            area = cv2.contourArea(cnt)
            if area > areaTH:

                #Tracking

                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)

                new = True
                if cy in range(gornja_granica,donja_granica):
                    for i in persons:
                        if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                            # objekat je blizak onom koji je vec otkriven
                            new = False
                            i.updateCoords(cx,cy)   #azuriranje koordinata u objektu
                            if i.going_UP(linija_dole,linija_gore) == True:
                                broj_gore += 1;
                                broj_ukupno += 1;
                                
                            elif i.going_DOWN(linija_dole,linija_gore) == True:
                                broj_dole += 1;
                                broj_ukupno += 1;
                                
                            break
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > donja_granica:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < gornja_granica:
                                i.setDone()
                        if i.timedOut():
                            #ukloni osobe sa liste
                            index = persons.index(i)
                            persons.pop(index)
                            del i     #oslobodi memoriju
                    if new == True:
                        p = Person.MyPerson(pid,cx,cy, max_p_age)
                        persons.append(p)
                        pid += 1     

               #Tracking slike
                cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
                
                
        
                
         
        # Crtaj slike
        
        str_up = 'UP: '+ str(broj_gore)
        str_down = 'DOWN: '+ str(broj_dole)
        frame = cv2.polylines(frame,[pts_L1],False,linija_dole_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L2],False,linija_gore_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        

        cv2.imshow('Frame',frame)
        
        
        #ESC za zatvaranje
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    #Upis u fajl
        
    cap.release()
    cv2.destroyAllWindows()
    f = open('result.txt','a+')
    f.write(str(broj_ukupno))
    f.write("\n")
    f.close()
    
