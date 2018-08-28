import cv2
import numpy as np
from PIL import ImageGrab
import time
from keras.models import load_model
from grabscreen import grab_screen
import pyautogui
from get_key import key_check
import os

#processing the image
def process_image(image):
	original_image=image
	processed_image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return processed_image

def screen_record():
        #load image from model
    model=load_model('dino-model.h5')
    #5 seconds timer
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time=time.time()
    while(True):
            #grab the region of the screen
            # 1. open your google chrome
            # 2. type "chrome://dino" in your address bar
            # 3. now select "windows key+left arrow" or similar key
            #    according to your os to align to extreme left of
            #    your window
            # 4. make sure that the little screen that popped up
            #    should be away from chrome browser
            
        screen=grab_screen(region=(60,270,280,300))
        print('took {}'.format(time.time()-last_time))
        last_time=time.time()
        #resize the screen
        newscreen=cv2.resize(screen,(50,50))
        #reshape the given array of screen so as to fit in the model
        #predict the outcome of the model
        pred=model.predict(newscreen.reshape(1,50,50,3))
        print(pred)
        if(pred>0):
                #jump
            pyautogui.press('space')
        cv2.imshow('window',newscreen)
        #press 'q' to exit the screen or press 'control+c'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
screen_record()
