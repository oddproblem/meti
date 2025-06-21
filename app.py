import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Digit Generator", layout="wide")
st.title("Digit Generator")
st.markdown("Select a digit below to generate 5 unique handwritten samples using our Conditional GAN.")

@st.cache_resource
def load_generator():
    return load_model("cgan_generator.h5")

generator = load_generator()
latent_dim = 100

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate"):
    noise = np.random.normal(0, 1, (5, latent_dim))
    labels = np.full((5, 1), digit)
    gen_imgs = generator.predict([noise, labels], verbose=0)
    gen_imgs = (gen_imgs + 1) / 2.0

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(gen_imgs[i, :, :, 0], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
