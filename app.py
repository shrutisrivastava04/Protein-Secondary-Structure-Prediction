import streamlit as st
from src.predict import predict_structure

st.set_page_config(page_title="Protein Predictor")
st.title("🧬 Protein Secondary Structure Predictor")
st.write("Enter a protein sequence below:")
sequence = st.text_area("Protein Sequence")

if st.button("Predict"):
    if sequence:
        result = predict_structure(sequence)
        st.subheader("Q3 Prediction")
        st.code(result['q3'])
        st.subheader("Q8 Prediction")
        st.code(result['q8'])
    else:
        st.warning("Please enter a sequence.")