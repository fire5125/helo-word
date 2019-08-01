#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:
import keras

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import cv2
from PIL import Image
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


model=load_model('my_prog.h5')


# In[4]:


os.chdir('cars')


# In[24]:


import cv2
import numpy as np
from PIL import Image

import os
os.getcwd()
alnums = []
k4=0
ld = os.listdir()
for i8 in ld:
    if i8.split('.')[-1]=='jpg':
        frame_name = i8
        im1 = cv2.imread(frame_name)
        #im2 = cv2.resize(im1, (500,500))
        im2 = im1
        cv2.imshow('1',im2)
        if cv2.waitKey():
            cv2.destroyAllWindows()
        #cv2.imwrite(('1'+str(k4)+'.jpg'),im2)


        # In[ ]:





        # In[33]:


        gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        blurred_im = cv2.blur(gray, (5,5))
        blurred_im1 = cv2.blur(blurred_im, (5,5))
        #blurred_im2 = cv2.blur(blurred_im1, (10,10))

        edged = cv2.Canny(blurred_im1,30,200)
        cv2.imshow('Canny edges', edged)
        if cv2.waitKey():
            cv2.destroyAllWindows()
        cv2.imwrite(('out/Canny edges'+str(k4)+'.jpg'),edged)
        


        # In[34]:


        _, contours, hierarchy = cv2.findContours( edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # отображаем контуры поверх изображения


        # In[35]:


        max1 = cv2.contourArea(contours[0])
        k = 0
        max2 = 0
        for i in contours:

            if cv2.contourArea(i) > max1:
                max1 = cv2.contourArea(i)
                max2 = k
            k += 1
     #   print(max2)


        # In[36]:


        IMAGE_kirill = Image.fromarray(im2)

        x2 = (max(contours[max2][:,0,0]))
        x1 = (min(contours[max2][:,0,0]))
        y2 = (max(contours[max2][:,0,1]))
        y1 = (min(contours[max2][:,0,1]))

        #im3 = cv2.getRectSubPix(im2, (x1, y1), (x2, y2))
        #cv2.imshow('Img', im3)
        #if cv2.waitKey():
        #    cv2.destroyAllWindows()


        # In[37]:



        from PIL import Image
        im2 = Image.open(frame_name)
        im3 = im2.crop((x1, y1, x2, y2))


        # In[38]:


        im3


        # In[39]:


        im4 = np.array(im3)
        im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)


        # In[40]:




        im5 = cv2.Canny(im4,30,200)
        cv2.imshow('Canny edges', im5)
        if cv2.waitKey():
            cv2.destroyAllWindows()
        cv2.imwrite(('out/Canny edges2'+str(k4)+'.jpg'),im5)

        _, contours, hierarchy = cv2.findContours( im5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        # In[41]:


        #_, contours, hierarchy = cv2.findContours( edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        # In[42]:


        max1 = cv2.contourArea(contours[0])
        k = 0
        max2 = 0
        for i in contours:

            if cv2.contourArea(i) > max1:
                max1 = cv2.contourArea(i)
                max2 = k
            k += 1
  #      print(max2)


        # In[43]:


        k=0
        p0 = []
        for i in contours:
            #print(cv2.contourArea(i))
            if cv2.contourArea(i)>50:
              #  print (cv2.contourArea(i))
                p0.append(k)
            k+=1



        # In[44]:


        p0


        # In[45]:


        k8=[]
        nums = []
        for i in p0[:-1]:
            im2 = Image.open(frame_name)
            im3 = im2.crop((x1, y1, x2, y2))


            x02 = (max(contours[i][:,0,0]))
            x01 = (min(contours[i][:,0,0]))
            y02 = (max(contours[i][:,0,1]))
            y01 = (min(contours[i][:,0,1]))


            im002 = im3.crop((x01-5, y01-5, x02+5, y02+5))
            im002 = im002.convert('LA')
            k8.append(im002)
            im004 = np.array(im002)
            img2 = cv2.resize(im004, (40,55))

            im01=np.array(img2)[:,:,0]
            #print(im01.shape)
            im0 = cv2.resize(im01, (40,55))
            #print(im0.shape)
            im0 = im0.reshape(1, 55, 40, 1)/255.0
            nums.append(int(model.predict_classes(im0)))
            #otvet.append
           # cv2.imwrite()
        k4+=1
        alnums.append(nums)


# In[28]:


print(alnums)


# In[27]:


for i2 in alnums:
    s1 = ''
    print('sled')
    for i3 in (i2):
        if i3 =='[' or i3 == ']':
            continue
        if int(i3) > 9:
            print('Это буква')
        else:
            print('Это цифра')
                


# In[ ]:




