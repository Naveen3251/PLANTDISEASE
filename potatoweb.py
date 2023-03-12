import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt



def predictpotato_class(image):
    with st.spinner('Loading Model...'):
        classname=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        model = keras.models.load_model(r'potato1.h5')
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        predicted_class = classname[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence

def predicttomato_class(image):
    with st.spinner('Loading Model...'):
        classname=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
        model = keras.models.load_model(r'tomato.h5')
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        predicted_class = classname[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence

def potato():
    st.title('Potato Leaf Disease Prediction')
    file_uploaded = st.file_uploader('Choose an image...', type = 'jpg',key=1)
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predictpotato_class(image)
        st.write('Prediction : {}'.format(result))
        st.write('Confidence : {}%'.format(confidence))

def tomato():
    st.title('Tomato Leaf Disease Prediction')
    file_uploaded = st.file_uploader('Choose an image...', type='jpg',key=2)
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predicttomato_class(image)
        st.write('Prediction : {}'.format(result))
        st.write('Confidence : {}%'.format(confidence))

def main():

    t= st.sidebar.radio('PLANT DISEASE',['POTATO DISEASE', 'TOMATO DISEASE'])
    if t== 'POTATO DISEASE':
        potato()
    if t == 'TOMATO DISEASE':
        tomato()
if __name__=="__main__":
    main()
