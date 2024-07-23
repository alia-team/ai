import streamlit as st
import os
from PIL import Image
from mlp import *
from cnn import *
# from naive_rbf import *
# from rbf import *
import data_processing as dp

def image_to_vector(image):
    image_array = np.array(image)
    image_vector = image_array.flatten()
    return image_vector

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

    img_resize = image.resize((100,100))
    img = img_resize.convert('P', palette=Image.ADAPTIVE, colors=256)
    if not os.path.exists('tmp_img'):
        os.makedirs('tmp_img')
    img.save('tmp_img/image.png')
    img_vector = image_to_vector(img)
    prediction = [0., 0., 0.]

    if option == 'MLP':
        model = load_mlp('../models/mlp.json')
        prediction = model.predict(img_vector, True)
    elif option == 'CNN':
         st.write("Loading model...")
         model = load_cnn('../models/cnn.json')
         st.write("Predicting...")
         prediction = model.predict('tmp_img/image.png')
    # elif option == 'Naive RBF':
    #     model = NaiveRBF()
    #     model.load_model('models/naive_rbf')
    #     prediction = model.predict(img_vector)
    # elif option == 'RBF':
    #     model = RBF()
    #     model.load_model('models/rbf')
    #     prediction = model.predict(img_vector)


    max_index = prediction.index(max(prediction))
    if max_index == 0:
        prediction = 'Avicularia'
    elif max_index == 1:
        prediction = 'Phidippus'
    elif max_index == 2:
        prediction = 'Tegenaire'
    else:
        prediction = 'error during prediction'

    

    os.remove('tmp_img/image.png')


    st.write(f'Prediction: {prediction}')




