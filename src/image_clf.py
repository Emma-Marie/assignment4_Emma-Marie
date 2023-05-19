import cv2
import pandas as pd
import os
import numpy as np
import argparse
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
# for freezing model parameters
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
# for predictions
from keras.utils import to_categorical
#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# for evaluation plot
import matplotlib.pyplot as plt
# utils plotting function
import sys
sys.path.append(".")
import utils.plotting as pl

def input_parse():
    #initialie the parser
    parser = argparse.ArgumentParser()
    # add argument
    parser.add_argument("--subfolder", type=str, default="Frescoes") # name of one of the four folders
    parser.add_argument("--image_name", type=str, default="fresco_1.jpg") # name of the the image you want to classify
    # parse the arguments from command line
    args = parser.parse_args()
    return args

def get_data():
    # getting data
    data_path = "in/Orthodox_Churches/"
    # list all classes
    class_labels = ["Chandelier","Dome","Frescoes","Lunette"]

    # create data frame of image path and label
    img_list = []
    label_list = []
    for label in class_labels:
        for img_file in os.listdir(data_path+label): # loop through all subfolders
            img_list.append(data_path+label+'/'+img_file) # add file location of all images to img_list
            label_list.append(label) # add label based on subfolder name
    df = pd.DataFrame({'img':img_list, 'label':label_list}) # create data frame
    print(df['label'].value_counts())

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

    return class_labels, X_train, y_train, X_test, y_test, X_val, y_val

def img_classifier(class_labels, X_train, y_train, X_val, y_val):
    # load model
    base_model = VGG16(input_shape=(96,96,3), include_top=False, weights='imagenet')

    # freeze model parameters (now only parameters in last layer can be adjusted)
    for layer in base_model.layers:
        layer.trainable = False
    base_model.layers[-2].trainable = True
    base_model.layers[-3].trainable = True
    base_model.layers[-4].trainable = True

    # add layers to model. The layers Flatten, Dropout, and Dense are added to VGG16
    model = Sequential()
    model.add(Input(shape=(96,96,3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(class_labels), activation='softmax'))
    model.summary()

    # Train model
    model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])
    history = model.fit(
        X_train, 
        y_train, 
        epochs=5, 
        validation_data=(X_val, y_val))

    return model, history

def predict_image(args, model, class_labels, y_test):
    folder = args.subfolder
    image_name = args.image_name
    # file path
    filepath = os.path.join("in", "Orthodox_Churches", folder, image_name)
    # load image
    img = load_img(filepath, target_size=(96, 96))
    # convert the image pixels to a numpy array
    image = img_to_array(img)
    # convert to rank 4 tensor
    image = np.expand_dims(image, axis=0)
    # prepare the image for the VGG model (Ross is not sure why we have to do this, but we do)
    image = preprocess_input(image)

    # Get the predicted probabilities
    y_pred = model.predict(image)
    # Convert y_test to one-hot encoded vectors
    num_classes = 4  # Replace with the actual number of classes in your classification problem
    y_test_encoded = to_categorical(y_test, num_classes)
    prediction_int =  y_pred.astype(int)# turn prediction from float into integer
    # get the predicted label
    predicted_class_index = np.argmax(prediction_int)
    img_label = class_labels[predicted_class_index]

    return img_label, y_test_encoded, y_test

def main():
    args = input_parse()
    # load, proces and split data
    class_labels, X_train, y_train, X_test, y_test, X_val, y_val = get_data()
    # train classifier model
    model, history = img_classifier(class_labels, X_train, y_train, X_val, y_val)
    # predict class of chosen image
    img_label, y_test_encoded, y_test = predict_image(args, model, class_labels, y_test)
    print(f"The target image is an image of a: {img_label}")

    # plot evaluation
    plt.plot(history.history['acc'], marker='o')
    plt.plot(history.history['val_acc'], marker='o')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    #Save plot
    output_path = os.path.join("out", "evaluation_acc_plot.png")
    plt.savefig(output_path, dpi = 100)
    print("Plot is saved!")

    # plot evaluation
    plt.plot(history.history['loss'], marker='o')
    plt.plot(history.history['val_loss'], marker='o')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'loss'], loc='lower right')
    #Save plot
    output_path = os.path.join("out", "evaluation_loss_plot.png")
    plt.savefig(output_path, dpi = 100)
    print("Plot is saved!")

    # Training and validation history plots
    pl.plot_history(history, 5)
    output_path = os.path.join("out", "train_and_val_plots.png")
    plt.savefig(output_path, dpi = 100)
    print("Plot is saved!")

    # predictions
    predictions = model.predict(X_test, batch_size=128)
    # classification report
    report = classification_report(y_test_encoded.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=class_labels)
    # Save report in "out"
    file_path = os.path.join("out", "classification_report.txt")
    with open(file_path, "w") as f: #"writing" classifier report and saving it
        f.write(report)
    print("Classification report is saved!")

if __name__ == "__main__":
    main()