import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import re

# Set page config
st.set_page_config(page_title="SkimLit", page_icon="🧠📝")

# Title
st.title("🧠 SkimLit — AI-Powered Text Classifier")
st.write(
    "Analyze scientific abstracts in seconds. "
    "**Background**, **Objective**, **Methods**, **Results**, and **Conclusions**."
)

# Load your saved model
def load_tensorflow_model():
    model = tf.keras.models.load_model("Skimlit.keras")
    return model
model  = load_tensorflow_model()

# Text input
abstract = st.text_area("Enter your abstract:", height=500)

if st.button("Classify"):
    if abstract.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentences = [s.strip() for s in re.split(r'(?<!\d)\.\s+', abstract) if s.strip()]
        total_lines = max(0, len(sentences) - 1)
        
        line_numbers = np.arange(len(sentences), dtype=np.int32).reshape(-1, 1)
        total_lines_arr = np.full((len(sentences), 1), total_lines, dtype=np.int32)

        text_tensor = tf.constant(sentences, dtype=tf.string)

        # Return dataset matching model input signature
        dataset = tf.data.Dataset.from_tensor_slices({
            "line_number_inputs": line_numbers,
            "total_number_inputs": total_lines_arr,
            "token_inputs_embedding": text_tensor
        }).batch(32).prefetch(tf.data.AUTOTUNE)

        class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

        labels = class_names

        predictions = model.predict(dataset)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = [class_names[i] for i in predicted_classes]
        result = dict(zip(sentences, predicted_labels))
        result = pd.DataFrame(list(result.items()), columns=['Sentence', 'Label'])


        st.subheader("Predictions:")
        st.write("---")
        for sentence, label in zip(result["Sentence"], result["Label"]):
            st.markdown(f"**{label}** → {sentence}")



