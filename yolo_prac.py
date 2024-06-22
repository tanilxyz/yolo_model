#!/usr/bin/env python
# coding: utf-8

# In[45]:


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')


# In[46]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[48]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt')


# In[49]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[50]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[51]:


model


# In[52]:


img = 'https://media.cntraveler.com/photos/53e2f41cdddaa35c30f66775/master/pass/highway-traffic.jpg'


# In[53]:


results = model(img)
results.print()


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[55]:


plt.imshow(np.squeeze(results.render()))


# In[56]:


cap = cv2.VideoCapture("C:/Users/tanil/yolov5/traffic.mp4")
while  cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        continue
    results = model(frame)
    if results is None:
        continue
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[57]:


cap = cv2.VideoCapture(0)
while  cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        continue
    results = model(frame)
    if results is None:
        continue
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[58]:


import uuid
import os
import time


# In[59]:


IMAGES_PATH = os.path.join("C:Users/tanil/yolo/pics", "C:/Users/tanil/yolo/pics/images")
labels = ['mouse', 'earphones']
number_imgs = 7


# In[20]:


cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('Image Collection', frame)
        time.sleep(2)
        if cv2.waitKey(10) & 0xFF == ('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[60]:


get_ipython().system('git clone https://github.com/HumanSignal/labelImg.git')


# In[95]:


get_ipython().system('pip install pyqt5 lxml --upgrade')
get_ipython().system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')


# In[97]:


get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2')


# In[104]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/last.pt', force_reload=True)


# In[105]:


img = os.path.join("C:Users/tanil/yolo/pics", "C:/Users/tanil/yolo/pics/images", 'mouse.460677e5-3097-11ef-826a-ca14833f7927.jpg')


# In[106]:


results = model(img)


# In[107]:


results.print()


# In[108]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show


# In[110]:


cap = cv2.VideoCapture(0)
while  cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        continue
    results = model(frame)
    if results is None:
        continue
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

