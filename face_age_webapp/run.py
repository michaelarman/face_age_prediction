from fastai.vision import open_image, load_learner, Image, image, torch
import torchvision.transforms as T
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO
import cv2

st.title("Age Predictor!")

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')

def detect_crop_face(img):
    img = PIL.Image.open(img).convert('RGB') 
    mywidth=200
    wpercent = (mywidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
    open_cv_image = np.array(img)  
    open_cv_image1 = open_cv_image[:, :, ::-1].copy() 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if len(faces) == 1:
        #st.markdown('Face Detected!')
        face_crop = []
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(open_cv_image1, (x,y), (x+w, y+h), (0,0,0), 3)
            # Define the region of interest in the image  
            face_crop.append(open_cv_image[y:y+h, x:x+w])
#         face_crop = np.array(face_crop)[0,:,:]
        face_crop = np.array(face_crop)[0,:,:]
        return face_crop
    # else: 
        #st.markdown("Uh-oh, couldn't detect a face")
    #     return False

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
    #   image = open_image(file_path)
    if detect_crop_face(file_path) is not None:
        img = detect_crop_face(file_path)
        img = PIL.Image.fromarray(img)
        img_tensor = T.ToTensor()(img)
        img_fastai = Image(img_tensor)
        # Get the image to display
        display_img = mpimg.imread(file_path)

        # Predict and display the image
        predict(img_fastai, display_img)
        predict_v2(img_fastai)
    else: 
        print('Try another photo')

if option == 'Choose your own image' and uploaded_file is not None:
    if detect_crop_face(uploaded_file) is not None:
        st.markdown('Face Detected!')
        img = detect_crop_face(uploaded_file)
        img = PIL.Image.fromarray(img)
        img_tensor = T.ToTensor()(img)
        img_fastai = Image(img_tensor)
        # Get the image to display
        #   pil_img = PIL.Image.open(uploaded_file)
        display_img = mpimg.imread(uploaded_file,0)

        # Predict and display the image
        predict(img_fastai, display_img)
        predict_v2(img_fastai)
    else:
        st.markdown("Uh-oh, couldn't detect a face or too many faces")
        st.markdown('Sorry, try another photo!')