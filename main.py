import cv2
import numpy as np
from PIL import ImageGrab
import time
from keras.models import load_model
from grabscreen import grab_screen
import pyautogui
import glob
import os


def screen_record():
        #load image from model
    model=load_model('dino-model.h5')
    print('make sure that your chrome dino is visible without any overlapping')
    dino=cv2.imread('dino.jpg',0)
    w_dino,h_dino=dino.shape[::-1]
    #5 seconds timer
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time=time.time()
    scr=grab_screen(region=(0,0,750,750))
    scr_gray=cv2.cvtColor(scr,cv2.COLOR_BGR2GRAY)
    res_dino=cv2.matchTemplate(scr_gray,dino, cv2.TM_CCOEFF_NORMED)
    threshold=0.8
    loc_dino=np.where(res_dino>=threshold)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res_dino)
    print(min_val,max_val,min_loc,max_loc)
    while(True):
            #grab the region of the screen
            # 1. open your google chrome
            # 2. type "chrome://dino" in your address bar
            # 3. make sure that nothing blocks the dino
        
        screen=grab_screen(region=((max_loc[0]+45),(max_loc[1]+17),(max_loc[0]+185),(max_loc[1]+42)))
        ''' uncomment to get the frame rate
        print('took {}'.format(time.time()-last_time))
        last_time=time.time()
        '''
        #resize the screen
        newscreen=cv2.resize(screen,(50,50))
        #reshape the given array of screen so as to fit in the model
        #predict the outcome of the model
        pred=model.predict(newscreen.reshape(1,50,50,3))
        ''' uncomment to get prediction result
        print(pred)
        '''
        if(pred>0):
                #jump
            pyautogui.press('space')
        #cv2.imshow('window',newscreen)
        #press 'q' to exit the screen or press 'control+c'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
screen_record()
