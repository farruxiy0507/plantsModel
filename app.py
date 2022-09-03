import streamlit as st
from fastai.vision.all import *
import plotly.express as px
st.title("O'simliklarni klassifikatsiya qiluvchi model")

#rasm joylash
file = st.file_uploader("Rasm yuklash", type=['jpeg', 'tiff', 'png', 'jpg'])




if file:
    st.image(file)
    #PIL conver
    img = PILImage.create(file)

    #modelni yuklab olamiz
    model = load_learner('plants_model.pkl')
    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)