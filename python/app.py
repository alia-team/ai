import streamlit as st
import os
from PIL import Image
# from mlp import *
from cnn import *
# from naive_rbf import *
# from rbf import *
import data_processing as dp

def convert_to_palette_mode_file(image):
    image = Image.open(image)
    image = image.convert('P', palette=Image.ADAPTIVE, colors=256)
    return image

def resize_image(image, size):
    image = Image.open(image)
    image = image.resize(size)
    return image

st.title('ALIA')

st.write('Bienvenu sur ALIA, l\'application qui vous permet d\'en apprendre plus sur les araignées et de savoir à laquelle vous avez a faire.')

option = st.selectbox(
    'Quel modèle souhaitez-vous utiliser ?',
    ('MLP', 'CNN', 'Naive RBF', 'RBF')
)

uploaded_file = st.file_uploader("Choisissez une image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = resize_image(uploaded_file, (100, 100))
    image = convert_to_palette_mode_file(uploaded_file)

    image.save('tmp_img/image.png')
    img_vector = dp.image_to_vector('tmp_img/image.png')

    # if option == 'MLP':
    #     model = MLP()
    #     model.load_model('models/mlp')
    #     prediction = model.predict(img_vector)
    if option == 'CNN':
        print("loading model...")
        model = load_cnn('../models/cnn.json')
        print("predicting...")
        prediction = model.predict('tmp_img/image')
        prediction = str(prediction)

    # elif option == 'Naive RBF':
    #     model = NaiveRBF()
    #     model.load_model('models/naive_rbf')
    #     prediction = model.predict(img_vector)
    # elif option == 'RBF':
    #     model = RBF()
    #     model.load_model('models/rbf')
    #     prediction = model.predict(img_vector)

    

    os.remove('tmp_img/image.png')

    # clean page
    st.empty()

    st.write('Done')

    st.write(f'Prediction: {prediction}')




