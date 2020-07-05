from fastai.vision import *
from fastai.vision import image
from fastai.basics import *
from fastai.callbacks import *
from fastai.widgets import ClassConfusion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO

st.title("Age Predictor!")


def predict(img, display_img):
        
        # Display the test image
        st.image(display_img, use_column_width=True)
    
        # Temporarily displays a message while executing 
        with st.spinner('Using model 1...'):
            time.sleep(3)
    
        # Load model and make prediction
        model = load_learner('models/', 'export.pkl')
        pred_class = model.predict(img)[0] # get the predicted class
        pred_prob = round(torch.max(model.predict(img)[2]).item()*100) # get the max probability
            
        # Display the prediction
        if str(pred_class) == '0-4':
            st.success("You are 0-4 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '5-9':
            st.success("You are 5-9 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '10-14':
            st.success("You are 10-14 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '15-19':
            st.success("You are 15-19 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '20-24':
            st.success("You are 20-24 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '25-29':
            st.success("You are 25-29 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '30-34':
            st.success("You are 30-34 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '35-39':
            st.success("You are 35-39 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '40-44':
            st.success("You are 40-44 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '45-49':
            st.success("You are 45-49 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '50-54':
            st.success("You are 50-54 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '55-59':
            st.success("You are 55-59 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '60-64':
            st.success("You are 60-64 with the probability of " + str(pred_prob) + '%.')
        else:
            st.success("You are 65+ with the probability of " + str(pred_prob) + '%.')

def predict_v2(img):
    
        # Temporarily displays a message while executing 
        with st.spinner('Using model 2...'):
            time.sleep(3)
    
        # Load model and make prediction
        model = load_learner('models/', 'export_v2.pkl')
        pred_class = model.predict(img)[0] # get the predicted class
        pred_prob = round(torch.max(model.predict(img)[2]).item()*100) # get the max probability
            
        # Display the prediction
        if str(pred_class) == '0-4':
            st.success("You are 0-4 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '5-9':
            st.success("You are 5-9 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '10-14':
            st.success("You are 10-14 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '15-19':
            st.success("You are 15-19 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '20-24':
            st.success("You are 20-24 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '25-29':
            st.success("You are 25-29 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '30-34':
            st.success("You are 30-34 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '35-39':
            st.success("You are 35-39 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '40-44':
            st.success("You are 40-44 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '45-49':
            st.success("You are 45-49 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '50-54':
            st.success("You are 50-54 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '55-59':
            st.success("You are 55-59 with the probability of " + str(pred_prob) + '%.')
        elif str(pred_class) == '60-64':
            st.success("You are 60-64 with the probability of " + str(pred_prob) + '%.')
        else:
            st.success("You are 65+ with the probability of " + str(pred_prob) + '%.')


option = st.radio('', ['Choose a test image', 'Choose your own image'])
uploaded_file = st.file_uploader("Choose an image")
if option == 'Choose a test image':
            
          # Test image selection
          test_images = os.listdir('test_images/')
          test_image = st.selectbox(
              'Please select a test image:', test_images)
            
          # Read the image
          file_path = 'test_images/' + test_image
          img = open_image(file_path)
            
          # Get the image to display
          display_img = mpimg.imread(file_path)
            
          # Predict and display the image
          predict(img, display_img)
          predict_v2(img)

if uploaded_file is not None:
        # display_img = mpimg.imread(uploaded_file,0)
        # size=200,200
        basewidth=200
        pil_img = PIL.Image.open(uploaded_file)
        wpercent = (basewidth/float(pil_img.size[0]))
        hsize = int((float(pil_img.size[1])*float(wpercent)))
        pil_img = pil_img.resize((basewidth,hsize), PIL.Image.BILINEAR).convert('RGB')
        # pil_img = pil_img.thumbnail(size)
        display_img = np.asarray(pil_img) # Image to display
        img = pil_img.convert('RGB')
        # img = img.thumbnail(size)
        img = image.pil2tensor(img, np.float32).div_(255)
        img = image.Image(img)
        predict(img, display_img)
        predict_v2(img)
        #   url = st.text_input("Please input a url:")
            
        #   if url != "":
        #       try:
        #           # Read image from the url
        #           response = requests.get(url)
        #           pil_img = PIL.Image.open(BytesIO(response.content))
        #           display_img = np.asarray(pil_img) # Image to display
            
        #           # Transform the image to feed into the model
        #           img = pil_img.convert('RGB')
        #           img = image.pil2tensor(img, np.float32).div_(255)
        #           img = image.Image(img)
            
        #           # Predict and display the image
        #           predict(img, display_img)
            
        #       except:
        #           st.text("Invalid url!")