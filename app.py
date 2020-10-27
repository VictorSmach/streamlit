import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.hdf5')
st.write("""
         # Bee health checker for Varroa MItes and state of bee health
         """
         )
st.write("It is an Image classification algorithm")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)) / 255

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    st.write(prediction)
