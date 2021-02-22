import streamlit as st
import random
import pandas as pd


st.title('Priorizador de andamentos')
st.subheader('Prova de conceito: T10 para NPL Brasil')

with st.spinner('Carregando modelo de classificaçao... 🧠'):
    from src import predict

text_input = st.text_area(label='Cole um andamento ou publicaçao, na íntegra:')

label_map = {
    0: 'baixa 🐸',
    1: 'media 🥎',
    2: 'alta 🚨'
}

if text_input:
    with st.spinner('Consultando os PMs...'):
        priority, confidence = predict.evaluate(text_input, confidence_score=True)
    
    st.write(f'Este andamento tem prioridade **{label_map[priority]}**')
    df = pd.DataFrame.from_dict(
        confidence, orient='index', columns=['confidence score']
    )
    df

