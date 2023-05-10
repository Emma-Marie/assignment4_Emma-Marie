# base tools
import os, sys
#sys.path.append(os.path.join(".."))

# data analysis
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm # gives a progress bar which is nice and easy to understand. 
# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)

# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# utils feature extraction
import sys
sys.path.append(".")
import utils.features as fe
# save dataframes
import pandas as pd

def get_data(model):
    #path to the whole datasets
    root_dir = os.path.join("in", "Orthodox_Churches", "Frescoes")
    filenames = [root_dir+"/"+name for name in sorted(os.listdir(root_dir))]

    # create list of features
    feature_list = []
    for i in tqdm(range(len(filenames)), position=0, leave=True):
        feature_list.append(fe.extract_features(filenames[i], model))

    return filenames, feature_list

def image_search(feature_list):
    # import NearestNeighbors
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=10, 
                                algorithm='brute',
                                metric='cosine').fit(feature_list)

    # Get features from chosen target image
    distances, indices = neighbors.kneighbors([feature_list[149]]) 
    # return list of the five closest images (excluding the image itself)
    most_similar = []
    for i in range(1,6):
        print(distances[0][i], indices[0][i])
        most_similar.append(indices[0][i])
    # Save list of the five closest images
    most_similar_df = pd.DataFrame(most_similar)
    df_path = os.path.join("out", "most_similar_metrix.csv")
    most_similar_df.to_csv(df_path, index=False)
    print("Df saved!")

    return most_similar

# Train classifier

def main():
    # load VGG16
    model = VGG16(weights='imagenet', 
                include_top=False, # Fals = doesn't include classifier layer
                pooling='avg',
                input_shape=(224, 224, 3))
    
    #load data and get list of features
    filenames, feature_list = get_data(model)
    print("Data and features are ready!")
    # get nearest neigbors
    most_similar = image_search(feature_list)
    print("Neighbors found")

    # plt target image and save
    plt.imshow(mpimg.imread(filenames[149]))
    outpath = os.path.join("out", "target_image.png")
    plt.savefig(outpath)
    # plot 5 most similar images and save
    f, axarr = plt.subplots(1,5)
    axarr[0].imshow(mpimg.imread(filenames[most_similar[0]]))
    axarr[1].imshow(mpimg.imread(filenames[most_similar[1]]))
    axarr[2].imshow(mpimg.imread(filenames[most_similar[2]]))
    axarr[3].imshow(mpimg.imread(filenames[most_similar[3]]))
    axarr[4].imshow(mpimg.imread(filenames[most_similar[4]]))
    outpath = os.path.join("out", "similar_images.png")
    plt.savefig(outpath)
    print("Plots saved!")


if __name__ == "__main__":
    main()