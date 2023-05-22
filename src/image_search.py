import os
import argparse
from tqdm import tqdm # gives a progress bar which is nice and easy to understand. 
# tensorflow
from tensorflow.keras.applications.vgg16 import (VGG16) 
# scikit-learn
from sklearn.neighbors import NearestNeighbors
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# utils feature extraction
import sys
sys.path.append(".")
import utils.features as fe
# for dataframe
import pandas as pd

def input_parse():
    #initialie the parser
    parser = argparse.ArgumentParser()
    # add argument
    parser.add_argument("--subfolder", type=str, default="Frescoes") # name of the selfchosen subfolder
    parser.add_argument("--number", type=int, default=0) # the number of the selfchosen target image
    # parse the arguments from command line
    args = parser.parse_args()
    return args

def get_data(args, model):
    #path to the whole datasets
    folder = args.subfolder
    root_dir = os.path.join("in", "Orthodox_Churches", folder)
    filenames = [root_dir+"/"+name for name in sorted(os.listdir(root_dir))]
    # create list of features
    feature_list = []
    for i in tqdm(range(len(filenames)), position=0, leave=True):
        feature_list.append(fe.extract_features(filenames[i], model))

    return filenames, feature_list, folder

def image_search(args, filenames, feature_list, folder):
    neighbors = NearestNeighbors(n_neighbors=10, 
                                algorithm='brute',
                                metric='cosine').fit(feature_list)
    # Get features from chosen target image
    target_image_number = args.number
    distances, indices = neighbors.kneighbors([feature_list[target_image_number]]) 
    # return list of the five closest images (excluding the image itself)
    most_similar = []
    for i in range(0,6): 
        print(distances[0][i], indices[0][i])
        most_similar.append(indices[0][i])

    # Create list of dictionaries with filename and distance
    similar_images = []
    for i, index in enumerate(most_similar[1:6]):
        similar_image = {
            'Filename': os.path.basename(filenames[index]),
            'Distance': distances[0][i+1]
        }
        similar_images.append(similar_image)
    # create dataframe from list of similar images
    similar_df = pd.DataFrame(similar_images)
    #save dataframe
    df_name = (f"nearn_distances_{os.path.basename(filenames[target_image_number])}.csv")
    df_path = os.path.join("out", df_name)
    similar_df.to_csv(df_path, index=False)
    print("Dataframe saved")

    return most_similar, target_image_number

def img_plotting(filenames, most_similar, target_image_number):
    # create a plot of the size 2x3
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    # setting title of plot
    fig.suptitle(f"Target {os.path.basename(filenames[target_image_number])} and the 5 most similar images")
    # plotting target image
    axs[0, 0].imshow(mpimg.imread(filenames[target_image_number]))
    axs[0, 0].set_title(f"Target Image {os.path.basename(filenames[target_image_number])}")
    # plotting five most similar images
    axs[0, 1].imshow(mpimg.imread(filenames[most_similar[1]]))
    axs[0, 1].set_title(f'Similar Image 1: {os.path.basename(filenames[most_similar[1]])}')
    axs[0, 2].imshow(mpimg.imread(filenames[most_similar[2]]))
    axs[0, 2].set_title(f'Similar Image 2: {os.path.basename(filenames[most_similar[2]])}')
    axs[1, 0].imshow(mpimg.imread(filenames[most_similar[3]]))
    axs[1, 0].set_title(f'Similar Image 3: {os.path.basename(filenames[most_similar[3]])}')
    axs[1, 1].imshow(mpimg.imread(filenames[most_similar[4]]))
    axs[1, 1].set_title(f'Similar Image 4: {os.path.basename(filenames[most_similar[4]])}')
    axs[1, 2].imshow(mpimg.imread(filenames[most_similar[5]]))
    axs[1, 2].set_title(f'Similar Image 5: {os.path.basename(filenames[most_similar[5]])}')
    # saving plot as a single image
    similar_plot_name = (f"nearn_similar_{os.path.basename(filenames[target_image_number])}.png")
    outpath = os.path.join("out", similar_plot_name)
    plt.savefig(outpath)
    print("Plot saved!")
    # plt target image and save
    #plt.imshow(mpimg.imread(filenames[target_image_number]))
    #target_plot_name = (f"target_image_{folder}{target_image_number}.png")
    #outpath = os.path.join("out", target_plot_name)
    #plt.savefig(outpath)
    # plot 5 most similar images and save
    #f, axarr = plt.subplots(1,5)
    #axarr[0].imshow(mpimg.imread(filenames[most_similar[0]]))
    #axarr[1].imshow(mpimg.imread(filenames[most_similar[1]]))
    #axarr[2].imshow(mpimg.imread(filenames[most_similar[2]]))
    #axarr[3].imshow(mpimg.imread(filenames[most_similar[3]]))
    #axarr[4].imshow(mpimg.imread(filenames[most_similar[4]]))
    #similar_plot_name = (f"most_similar_{folder}{target_image_number}.png")
    #outpath = os.path.join("out", similar_plot_name)
    #plt.savefig(outpath)
    #print("Plots saved!")

def main():
    # load VGG16
    model = VGG16(weights='imagenet', 
                include_top=False, # Fals = doesn't include classifier layer
                pooling='avg',
                input_shape=(224, 224, 3))
    # get parsed arguments
    args = input_parse()
    # load data and get list of features
    filenames, feature_list, folder = get_data(args, model)
    print("Data and features are ready!")
    # get nearest neigbors
    most_similar, target_image_number = image_search(args, filenames, feature_list, folder)
    print("Neighbors found!")
    # plot target image ans 5 most similar
    img_plotting(filenames, most_similar, target_image_number)

if __name__ == "__main__":
    main()