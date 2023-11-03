from string import punctuation
import streamlit as st
import joblib
import time
import cv2
import numpy as np

# Load the model
clf = joblib.load('image_processing.pkl')
# vgg = joblib.load('embeddings.pkl')
pca = joblib.load('pca_model.pkl')
scaler = joblib.load('scaler_model.pkl')
le = joblib.load('le_model.pkl')
def change_photo_state():
    st.session_state["photo"] = "done"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Model

def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights('/Users/wongyenchik/Desktop/Image processing/vgg_face_weights.h5')
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face_descriptor

def preprocessing(uploaded_photo):
    # Read the uploaded image
    new_img = cv2.imdecode(np.frombuffer(uploaded_photo.read(), np.uint8), 1)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    new_img = cv2.resize(new_img, (224, 224))  # Resize the image to (224, 224)

    # Normalize the image and generate embedding
    new_img = (new_img / 255.).astype(np.float32)
    vgg = vgg_face()
    new_embedding = vgg.predict(np.expand_dims(new_img, axis=0))[0]

    # Standardize and apply PCA transformation
    new_embedding_std = scaler.transform(new_embedding.reshape(1, -1))

    new_embedding_pca = pca.transform(new_embedding_std)

    # Predict the label
    predicted_label = clf.predict(new_embedding_pca)

    predicted_class = le.inverse_transform(predicted_label)[0]

    return predicted_class

##=================================
# Set the page configuration
st.set_page_config(page_title="Face Recognition", page_icon="💃")

# Set the title and description
st.markdown("<h1 style='text-align: center; font-size: 4vw;'>Face Recognition System 💃</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>Explore our Face Recognition System: Dive into the world of fame with our cutting-edge technology that recognizes and identifies face effortlessly.</p>", unsafe_allow_html=True)
st.markdown("<h2></h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 15px;'>Only applicable for Bill Gates, Gal Gadot, Anne Hathaway, Emma Watson, Cristiano Ronaldo, Taylor Swift and Chris Hemsworth face.</p>", unsafe_allow_html=True)

if "photo" not in st.session_state:
    st.session_state["photo"] = "not done"

uploaded_photo = st.file_uploader("Upload a face photo", on_change=change_photo_state)

if st.session_state["photo"] == "done":
    progress_bar = st.progress(0)

    for perc_completed in range(100):
        time.sleep(0.05)
        progress_bar.progress(perc_completed+1)

    st.success("Photo uploaded successfully!")
    st.write("Here is the result.")
    st.image(uploaded_photo)
    # new_img = cv2.imdecode(np.fromstring(uploaded_photo.read(), np.uint8), 1)

    # Predict the label for the uploaded image
    predicted_class = preprocessing(uploaded_photo)
    st.write("Predicted Celebrity: **" + predicted_class + "**")



