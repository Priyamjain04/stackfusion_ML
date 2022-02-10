#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import urllib
import matplotlib.pyplot as plt
import cv2
import glob
import os
import time
from PIL import Image


# In[3]:


get_ipython().system('pip install opendatasets')


# In[4]:


import opendatasets as od


# In[5]:


od.download("https://www.kaggle.com/dataturks/vehicle-number-plate-detection/Indian_Number_plates")


# In[6]:


df = pd.read_json("/kaggle/input/datatruks/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)
df.head()


# In[7]:


df.shape


# In[8]:


new_csv=df.to_csv("indian_license_plates.csv", index=False)


# In[9]:


new_df=pd.read_csv("/kaggle/working/indian_license_plates.csv")
new_df.head(10)


# In[10]:


df['annotation'][0]


# In[11]:


os.mkdir("Number Plates")


# In[12]:


data = dict()
data["img_name"] = list()
data["img_width"] = list()
data["img_height"] = list()
data["top-x"] = list()
data["top-y"] = list()
data["bottom-x"] = list()
data["bottom-y"] = list()


# In[13]:


df['annotation'][0]


# In[14]:


df['annotation'][0][0]["points"]


# In[15]:


new_df.head(5)


# In[16]:


count = 0
for index, row in df.iterrows():
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert('RGB')
    img.save("Number Plates/car{}.jpeg".format(count), "JPEG")
    
    data["img_name"].append("car{}".format(count))
    
    d = row["annotation"]
    
    data["img_width"].append(d[0]["imageWidth"])
    data["img_height"].append(d[0]["imageHeight"])
    data["top-x"].append(d[0]["points"][0]["x"])
    data["top-y"].append(d[0]["points"][0]["y"])
    data["bottom-x"].append(d[0]["points"][1]["x"])
    data["bottom-y"].append(d[0]["points"][1]["y"])
    
    count += 1
    
print("Done Successfully")    


# In[17]:


new_data=pd.DataFrame(data)
new_data.head()


# In[18]:


new_data.shape


# In[19]:


new_data.describe()


# In[20]:


new_data['img_name']=new_data['img_name']+".jpeg"


# In[21]:


width= 300
height= 300
channels= 3


# In[22]:


def viewimage(t):
    
    image = cv2.imread("Number Plates/" + new_data["img_name"].iloc[t])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(width,height))
    
    top_x=int(new_data['top-x'].iloc[t]* width)
    top_y=int(new_data['top-y'].iloc[t]*height)
    bot_x=int(new_data['bottom-x'].iloc[t]*height)
    bot_y=int(new_data['bottom-y'].iloc[t]*height)
    
    
    new_img=cv2.rectangle(image,(top_x,top_y),(bot_x,bot_y),(0, 0, 255), 1)
    
    plt.imshow(new_img)
    
    plt.show()


# In[23]:


viewimage(10)


# In[24]:


viewimage(100)


# In[25]:


viewimage(80)


# In[26]:


from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


# In[27]:


tagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_generator = datagen.flow_from_dataframe(
    df_subset,
    directory="Number Plates/",
    x_col="img_name",
    y_col=["top-x", "top-y", "bottom-x", "bottom-y"],
    target_size=(width,height),
    batch_size=32, 
    class_mode="raw",
    subset="training")

validation_generator = datagen.flow_from_dataframe(
    df_subset,
    directory="Number Plates/",
    x_col="img_name",
    y_col=["top-x", "top-y", "bottom-x", "bottom-y"],
    target_size=(width,height),
    batch_size=32, 
    class_mode="raw",
    subset="validation")


# In[ ]:


model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(width,height,channels)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False
model.summary()


# In[28]:


adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss="mse",metrics=['accuracy'])
history = model.fit_generator(train_generator,
    steps_per_epoch=7,
    validation_data=validation_generator,
    validation_steps=1,
    epochs=20)


# In[29]:


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


# In[30]:


get_ipython().system('pip install easyocr')


# In[31]:


get_ipython().system('pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html')


# In[32]:


import easyocr


# In[33]:


detection_threshold = 0.7


# In[35]:


for idx, row in new_data.iloc[drop_indices].iterrows():    
    
    img = cv2.resize(cv2.imread("Number Plates/" + row['img_name']) / 255.0, dsize=(width,height))
    y_hat = model.predict(img.reshape(1, width,height, 3)).reshape(-1) * width
    
    xt, yt = y_hat[0], y_hat[1]
    xb, yb = y_hat[2], y_hat[3]
    
    img = cv2.cvtColor(img.astype(np.float32),cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(img, (xt, yt), (xb, yb), (255,0,255), 1)
    
    clone = image.copy() 
    
    # Cropping the predicted reactangle region into a new image
    crop_img = clone[int(yt):int(yb),int(xt):int(xb)] 
   
    plt.imshow(crop_img)
    im = Image.fromarray((crop_img * 255).astype(np.uint8))
    
    
   
 plt.show()


# In[36]:


reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
result


# In[37]:


text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


# In[38]:


print("Number plate recognised")


# In[ ]:




