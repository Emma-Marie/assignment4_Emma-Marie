# assignment4_vis_self-assigned_Emma-Marie

## 1. Contribution
The code for this assignment has been developed on my own, but with inspiration from classmates during the class lectures. The functions in the scripts ```features.py``` and ```plotting.py``` in the ```utils``` folder is provided by my teacher Ross. 

## 2. Description and purpose
This assignment consists of two main scripts: ```image_search.py``` is an image search script using k-nearest neighbor from ```Scikit-learn``` and the CNN VGG16 model from ```tensorflow.keras```, while ```image_clf.py``` is an image classification script, which uses a finetuned version of the VGG16 model to classify the image dataset. The assignment has two purposes:
1)	to train an image search model to find and plot the five images most similar to a chosen target image, which is parsed into the script using argparse. 
2)	To train an image classifier using a version of VGG16 of which the last layers are finetuned on this specific dataset. Then the model is able to predict the class of the target image chosen through argparse, and the class is printed in the terminal. 

## 3. Methods
My assignment consists of two scripts, which can both be found in the ```src``` folder:
The ```image_search.py``` load the VGG16 model and uses it together with the function from the ```features.py```utils script to get the image features for all images in the chosen subfolder and append them to a list. Then it gets the features from the chosen target image and uses ```k-nearest neighbor``` to find the most similar images to the target image and create a list of the five images most similar to the target image. The output is a data frame with the names and distance scores of these images, and a plot of these images and the target image. They are saved in the ```out``` folder. 
The ```image_clf.py``` creates a data frame of image paths and labels, creates a dictionary with the labels represented as numbers, combine the images and encoded labels, and split the data into train, test and validation data. Then the ```VGG16``` model is loaded and flatten, dropout and dense layers are added to the model, which is then trained on the data. The class of the image chosen through argparse is predicted and printed in the terminal. The output is a classification report and a plot showing the loss and accuracy curve of the train and validation data. They are saved in the ```out``` folder. The trained model is saved in the ```models``` folder. Note that the saved model hasn’t been pushed to GitHub because of the size limits on GitHub.  

### 3.1 Utils
The ```features.py``` script provided by Ross uses the function preprocessed_input() from VGG16  to extract image features, which are then flattened and normalized. 

## 4. Data
The data used for this assignment is called ```Cultural Heritage Dataset – Orthodox Churches``` and is downloaded from Kaggle. The dataset contains images of Christian Orthodox churches and consists of a main folder called ```Orthodox_Churches``` containing the four subfolders ```Chandelier```, ```Dome```, ```Frescoes```, and ```Lunette```. Each subfolder name is also the name of the class of the images, which it contains. Each class holds 200 images, which gives 800 images in total. 

### 4.1 Get the data
Before running the scripts, please download the dataset from Kaggle (4MB) here: https://www.kaggle.com/datasets/rjankovic/cultural-heritage-orthodox-churches. Place the data in the ```in``` folder so your folder structure looks like this:

```
- in
    - Orthodox_Churches
        - Chandelier
        - Dome
        - Frescoes
        - Lunette
```

## 5. Usage 

### 5.1 Prerequisites
Before running the scripts, you need to install Python 3 and Bash. I created and ran the code using the app “Coder Python 1.73.1” on Ucloud.sdu.dk. Clone the GitHub repository on your device to get started. 

### 5.2 Install packages
First step is to run the command “setup.sh“ through the command line to create a virtual environment and install the packages in requirements.txt:
            
            bash setup.sh

Second step is to activate the virtual environment manually by running “source ./image_search_env/bin/activate”:

            source ./image_search_env/bin/activate

### 5.3 Run the scripts
You run image_search.py by running the command “python3 src/image_search.py --subfolder --image” from the command line. The --subfolder argument is the name of one of the four subfolders Chandelier, Dome, Frescoes, or Lunette. The target image is found in this subfolder. If no subfolder is specified, “Frescoes” is the default folder. The --image argument is the name of the target image. If no image name is chosen, “fresco_1.jpg” is the default image. This image is also used in the code below:
    
                python3 src/image_search.py --subfolder Frescoes --image fresco_1.jpg

You run image_clf.py by running the command “python3 src/image_clf.py --subfolder --image” from the command line. The --subfolder argument is the name of one of the four subfolders Chandelier, Dome, Frescoes, or Lunette. The target image is found in this subfolder. If no subfolder is specified, “Frescoes” is the default folder. The --image argument refers to the name of the target image. If no image name is chosen, “fresco_1.jpg” is the default image. This image is also used in the code below:

                python3 src/image_clf.py --subfolder Frescoes --image fresco_1.jpg

## 6. Discussing of results
The accuracy of the model is 95% which is very high. This indicates that the model is overfitted, which means that it performs very well on the images in this dataset, but it wouldn’t be good at generalizing. Therefore, it would have a hard time recognizing images of frescoes, domes, chandeliers or lunettes, which it hasn’t seen before. Looking at the plots, the two loss curves should have started at a high loss in the beginning, then they should have gradually decreased before they evened out in the end. Instead, the curves are going up and down and don’t even out towards the end. This indicates that the learning process of the model isn’t optimal and that the model is overfit. Looking at the accuracy plot, the gap between the train and validation curves also indicates that the model is overfit. The overfitting can be due to the fact that The Cultural Heritage Dataset is quite small, and that the model is too complex for such a small dataset. 

## 7. References
RJankovic, “Cultural Heritage Dataset – Orthodox Churches”, Kaggle: https://www.kaggle.com/datasets/rjankovic/cultural-heritage-orthodox-churches (last visited May 19 2023). 

