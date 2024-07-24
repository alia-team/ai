import streamlit as st
import os
from PIL import Image
from mlp import *
from cnn import *
# from naive_rbf import *
# from rbf import *
import data_processing as dp

def image_to_vector(image: Image):
    image_array = np.array(image)
    image_vector = image_array.flatten()
    return image_vector
 
st.title('ALIA')
st.write('Welcome on ALIA, the app teaching you more about spiders and telling you which one hurts!')

option = st.selectbox(
    'Which model do you want to use?',
    ('MLP', 'CNN', 'Naive RBF', 'RBF')
)

uploaded_file = st.file_uploader("Upload an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploaded.', use_column_width=True)
    resized_img: Image = image.resize((100, 100))
    standardized_img: Image = resized_img.convert('P', palette=Image.ADAPTIVE, colors=256)
    standardized_img.save('tmp_img/image.png')

    prediction = [0.,0.,0.]

    if option == 'MLP':
        st.write("Loading model...")
        model = load_mlp('../models/mlp_best.json')

        st.write("Predicting...")
        img = Image.open('tmp_img/image.png')
        inputs = image_to_vector(img)
        image_mean = inputs.mean()
        image_std = inputs.std()
        inputs = [(i - image_mean) / image_std for i in inputs]
        prediction = model.predict(inputs, True)


    elif option == 'CNN':
        st.write("Loading model...")
        model = load_cnn('../models/cnn_240.json')

        st.write("Predicting...")
        if not os.path.exists('tmp_img'):
            os.makedirs('tmp_img')

        prediction = model.predict('tmp_img/image.png')

    # elif option == 'Naive RBF':
    #     st.write("Loading model...")
    #     model = NaiveRBF()
    #     model.load_model('models/naive_rbf')
    #
    #     st.write("Predicting...")
    #     prediction = model.predict(img_vector)
    #
    # elif option == 'RBF':
    #     st.write("Loading model...")
    #     model = RBF()
    #     model.load_model('models/rbf')
    #
    #     st.write("Predicting...")
    #     prediction = model.predict(img_vector)


    os.remove('tmp_img/image.png')
    print(prediction)

    max_index = prediction.index(max(prediction))

    if max_index == 0:
        prediction = 'Avicularia\n safe but painful bite'
    elif max_index == 1:
        prediction = 'Phidippus\n safe'
    elif max_index == 2:
        prediction = 'Tegenaria\n safe'
    else:
        prediction = 'Error during prediction.'


    st.write(f'Prediction: {prediction}')