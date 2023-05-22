# assignment4_vis_self-assigned_Emma-Marie

## 1. Contribution
I have created this assignment by myself and not in colaboration with other students. The classification part of the assignment is inspired by the blogpost "Image Classification for beginner" from MLearning.ai. 

## 2. Description and purpose

This assignment consists of two main scripts: an image search script (```image_search.py```) using the NearestNeighbors approach from Scikit-learn, and an image classification script (```image_clf.py```) using a finetuned version of VGG16 from tensorflow.keras. The two scripts are located in the ```src``` folder. 

The purpose of the assignment is the following:
- train an image search model using NearestNeighbors and VGG16.
    - find and plot the five images, which is most similar to a self-chosen image from the data set which is argparsed into the model through the commandline. 
    - save a plot with the target image and the plot of the five most similar images in the ```out``` folder. 
- train an image classifier using VGG16 and finetuning its last layers. 
    - create a classification report and save it in the ```out``` folder. 
    - create a validation and loss plot and save it in the ```out``` folder.
    - save the trained model in the ```models``` folder.
        - NB: No model is pushed to the ```models``` folder on Github because of the size of the model. 

## 3. Methods

## 4. Data
The data used in this assignment is called ```Cultural Heritage Dataset - Orthodox Churches``` and is downloaded from Kaggle. 

DESCRIBE DATA

### 4.1 Load the data
Before running the scripts, please download the dataset from Kaggle (4MB) here: https://www.kaggle.com/datasets/rjankovic/cultural-heritage-orthodox-churches. Place the data in the ```in``` folder, so your folder structure looks like this:

```
- in
    - Orthodox_Churches
        - Chandelier
        - Dome
        - Frescoes
        - Lunette
```

## 5. Usage and reproducibility 

### 5.1 Prerequisites
You must install xxx to be able to run the scripts. I have created and tested the scripts using Code Python 1.73.1 on UCloud.sdu.dk. 

### 5.2 Install packages
First step is to run the setup.sh script through the commandline to create a virtual environment and install the requirements.txt:
            
            bash setup.sh

Second step is to activate the virtual environment manually by running "source ./image_search_env/bin/activate" from the command line:

            source ./image_search_env/bin/activate

### 5.3 Running the scripts
You run the two scripts in the following way: 
- Running ```image_search.py```:
    Type "python3 src/image_search.py --subfolder --number". 
    - The subfolder argument is the name of one of the four subfolers Chandelier, Dome, Frescoes, and Lunette, from which the target image should be found. If no subfolder is chosen, "Frescoes" is the default folder.
    - The number argument is refering to the image, which you want as the target image. The number is not the image name, but refers to the order in which the images lay in the folder. If no image number is chosen, 0 is the default number. 
    The comment below is an random example with image number 10 in the Dome folder as the target image:
    
                python3 src/image_search.py --subfolder Dome --number 10

- Running ```image_clf.py```:
    Type "python3 src/image_clf.py --subfolder --image_name". 
    - The *subfolder* argument is the name of one of the four subfolers Chandelier, Dome, Frescoes, and Lunette, from which the target image should be found. 
    - The *image_name* argument is refering to the name of the image, which you want as the target image. If no subfolder is chosen, "fresco_1.jpg" is the default image name. 
    The comment below is an random example with the image called "dome_1.jpg" in the Dome folder:

                python3 src/image_clf.py --subfolder Dome --image_name dome_1.jpg

## Utils
In the ```utils``` folder is a script called ```features.py```. The script contains a function that extracts features from image data using pretrained model (e.g. VGG16). The function is created by my teacher Ross. The function from this script is called in my main script. 

## Discussing of the results

- High accuracy, but overfit model
- some classes has high accuracy and some very low

## References
Kimnaruk, Yannawut (June 4 2022), "Image Classification for beginner", MLearning.ai:  https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78 (last visited May 17 2023)

RJankovic, "Cultural Heritage Dataset - Orthodox Churches", Kaggle: https://www.kaggle.com/datasets/rjankovic/cultural-heritage-orthodox-churches (last visited May 19 2023)

