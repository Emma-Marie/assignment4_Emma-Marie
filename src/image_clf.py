import cv2
import pandas as pd
import os
import numpy as np
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# getting data
data_path = "in/Orthodox_Churches/"
# list all classes
class_label = ["Chandelier","Dome","Frescoes","Lunette"]

# create data frame of image path and label
img_list = []
label_list = []
for label in class_label:
    for img_file in os.listdir(data_path+label): # loop through all subfolders
        img_list.append(data_path+label+'/'+img_file) # add file location of all images to img_list
        label_list.append(label) # add label based on subfolder name
df = pd.DataFrame({'img':img_list, 'label':label_list}) # create data frame
print(df['label'].value_counts())

# Data preprocessing
# Create a dataframe for mapping label
df_labels = { # create dictionary with labels as numbers
    "Chandelier" : 0,
    "Dome" : 1,
    "Frescoes" : 2,
    "Lunette" : 3
}
# Encode
df['encode_label'] = df['label'].map(df_labels)

# Prepare training data set 
X = []
y = []
for img, label in zip(df['img'], df['encode_label']):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    X.append(img)
    y.append(label)
# Convert X and y to NumPy arrays
X = np.array(X)
y = np.array(y)

# train-validation-test split
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val) # the split is train 80%, validation for model tuning 4%, test set 16%

###CREATE MODEL
base_model = VGG16(input_shape=(96,96,3), include_top=False, weights='imagenet')

# freeze model parameters (now only parameters in last layer can be adjusted)
for layer in base_model.layers:
    layer.trainable = False
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# add layers to model. The layers Flatten, Dropout, and Dense are added to VGG16
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
model = Sequential()
model.add(Input(shape=(96,96,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(class_label), activation='softmax'))
model.summary()