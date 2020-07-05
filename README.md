# face_age_prediction

## Installation
All required packages can be found [here](https://github.com/michaelarman/face_age_prediction/blob/master/requirements.txt)

## Project Motivation
In this project, I wanted to use Convulutional Neural Networks (CNNs) to classify the age using face images. The easiest and fastest implementation of the CNN was through using FastAI. 
I leveraged transfer learning to train my model faster. I also performed data augmentations/transformations of the images to obtain a better generalization.
I formed 14 age brackets as the outputs:<br>
0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, and 65+ <br>
After training the models, I wanted to create a webapp using streamlit so people can try it out by using test images or upload images of their own.

## [Webapp]()
### Instructions for Running Locally 
Open the [faceapp folder](https://github.com/michaelarman/face_age_prediction/tree/master/face_age_webapp) and in your python terminal go to the path and run:
`streamlit run run.py`

## Datasets
There were 3 datasets that I tried to use.
- [UTKFace](https://www.kaggle.com/jangedoo/utkface-new?) - This was the cleanest of the datasets. In some of the ages, there were some errors but was the most structured. ~30,000 images
- [IMDb](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) - Large dataset of images of celebrities. A lot are incorrectly labelled or broken ~ 460k images
- [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) - Less messy than IMDb but not the cleanest either ~62k

## Challenges
Both the IMDb and Wiki datasets were very messy and some images were incorrectly labelled or broken. 
Moreoever, the faces in the images were not aligned or cropped as good as the UTKFace dataset. This made incorporating the images difficult and raises the question:
Is more data better, even in the case that it can be incorrectly labelled or unstructured?
Due to this problem, I attempted 3 versions of the CNN.
1. Only using the UTKFace dataset but manually deleteed images that were incorrectly labelled (which wasn't much). The number of images used were ~19,000
2. Incorporating some IMDb and Wiki images but manually adding the images that seem correctly labelled and not broken.
3. Using all of the data

## Results
1. Approach v1 achieved over 80% accuracy and if trained for more epochs, probably would've achieved a better accuracy. 
Data Transforms were the standard transforms that FastAI's `get_transform` uses. 
The pretrained model that was used to train on was resnet50's architecture
This model performs well for predicting face images that are cropped and aligned perfectly. 
Although it attained a great accuracy score with this data, it is not very good at predicting new images that aren't structured perfectly. 
This can be seen in the webapp.
2. Approach v2 did not perform as well. It achieved a 
Data Transforms were the standard transforms that FastAI's `get_transform` uses and also random_resize_crop since not all of the images were properly cropped or aligned.
The pretrained model that was used to train on was mobilenet_v2's architecture


