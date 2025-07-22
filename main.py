import streamlit as st
import tensorflow as tf
import numpy as np

# tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('models/CNN.keras')

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0

    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)
#    confidence = prediction[0][0]

#    st.write(f'Faw model pprediction: {confidence:.4f}')
#    return 'Cancerous' if confidence > 0.5 else 'Not Cancerous'
    return 'Cancerous' if prediction[0][0] > 0.5 else 'Not Cancerous'

# UI sidebar
st.sidebar.title('Colon Cancer CNN')
app_mode = st.sidebar.selectbox('Pages', ['Cancer Predictor', 'About'])

# UI cancer predictor
if(app_mode=='Cancer Predictor'):
    st.header('Cancer Predictor')
    test_image = st.file_uploader('Upload an Image')
    if(st.button('Display Uploaded Image')):
        st.image(test_image, use_container_width=True)
    if(st.button('Predict Cancer')):
        st.write('CNN Model Prediction')
        result = model_prediction(test_image)
        st.success('CNN Model Prediction: {}'.format(result))

# UI about
if(app_mode=='About'):
    st.header('About')
    st.markdown("""
    ### Cancer Predictor CNN
    Goal: Build a deep learning model using a Convolutional Neural Netowrk (CNN) to classify histopathological images of colon tissue as cancerous or non-cancerous.

    Dataset: [Lung and Colon Cancer Histopathological Images (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

    Output: A trained CNN model capable of accurately predicting new histopathologiical images as cancerous or non-cancerous.

    Evaluation Metrics: Accuracy, Confusion Matrix
    """
    )