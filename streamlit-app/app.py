import streamlit as st
from model_helper import predict
from pathlib import Path

st.set_page_config(layout="wide")

with open(Path(__file__).parent / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    with col2:
        prediction = predict(image_path)
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <p style="font-size: 3rem; font-weight: bold; text-align: center;">
                    Predicted Class:<br>{prediction}
                </p>
            </div>
        """, unsafe_allow_html=True)
