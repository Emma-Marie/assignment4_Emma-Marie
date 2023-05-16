# assignment4_vis_self-assigned_Emma-Marie

## Description and purpose

Dataframe with the number of the most similar images is saved in ```out``` folder. The df shows:
- first column is the distances (the cosin similarity)
- secon column is indices (the nearest neighbours)
NO, BUT I WISH:)

## Data
The data used in this assignment is called ```Cultural Heritage Dataset - Orthodox Churches``` and is downloaded from Kaggle. 

### How to load the data

## Script

The script is called ```image_search.py``` and is lokated in the ```src``` folder. 

## How to run the scripts

### Prerequisites

### Running the scripts

First step is to run the setup.sh script through the commandline to create a virtual environment and install the requirements.txt:

            bash setup.sh

Second step is to activate the virtual environment manually by running "source ./image_search_env/bin/activate" from the command line:

            source ./image_search_env/bin/activate
            
The third step is to run the text_generator.py from the commandline by typing "python3 src/image_search.py --subfolder --number". 
- The subfolder argument is the name of one of the four subfolers Chandelier, Dome, Frescoes, and Lunette, from which the target image should be found. 
- The number argument is refering to the image, which you want as the target image. The number is not the image name, but refers to the order in which the images lay in the folder. If no subfolder is chosen, "Frescoes" is set at the default folder. 
The comment below is an example with image number 10 in the Dome folder as the target image:

            python3 src/image_search.py --subfolder Dome --number 10

## Utils
In the ```utils``` folder is a script called ```features.py```. The script contains a function that extracts features from image data using pretrained model (e.g. VGG16). The function is created by my teacher Ross. The function from this script is called in my main script. 

## Discussing of the results

## References

Guide for the classification: https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78 (INCLUDE???)