import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Charger le modèle MobileNetV2 pré-entraîné
mobile_net_model = MobileNetV2(weights='imagenet')

st.image("http://www.ehtp.ac.ma/images/lo.png")
st.write("""
# MSDE5 : Projet Deep Learning
## Prédiction de la classification d'une image 
## Cat or Dog ?

Cette application permette la **Classification** d'une image selon un modèle CNN avec un **Transfer learning**.
""")

st.write("""
# Projet réalisé en binome par :
# Mohamed OMARI, El Houcine MSELLEK
""")

st.sidebar.image("https://img.buzzfeed.com/buzzfeed-static/static/2018-11/14/16/tmp/buzzfeed-prod-web-02/tmp-name-2-27675-1542232746-5_dblbig.jpg?resize=1200:*",width=450)

st.sidebar.header("Is it cat or dog ?")


# Fonction pour effectuer la prédiction sur une image
def predict(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = mobile_net_model.predict(image_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0][0][1]

# Interface utilisateur Streamlit
st.title("Image Classifier")

uploaded_file = st.file_uploader("Veuillez inserer une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Chargement de l'image.", use_column_width=True)

    # Effectuer la prédiction
    result = predict(image)

    # Afficher le résultat
    st.write("Prédiction du modèle :", result)
    #if result.lower() == 'cat':
    #    label = 0
    #elif result.lower() == 'dog':
    #    label = 1
    #else:
    #    label = -1  # Autre label

    #st.write("Label:", label)
