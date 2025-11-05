# 🧠 SkimLit Reimagined — Sequential Sentence Classification for Medical Abstracts

> A modern deep learning reimplementation of SkimLit — now rebuilt with TensorFlow 2 and deployed using Streamlit.  
> Classifies each sentence of a medical abstract into categories such as **BACKGROUND**, **OBJECTIVE**, **METHODS**, **RESULTS**, and **CONCLUSIONS**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25-success)

---

## 🎯 Problem Statement

Medical abstracts contain crucial information, but manually identifying sections like *Background* or *Results* slows down research.  
**SkimLit Reimagined** automates this process using machine learning to classify sentences, helping researchers and professionals **skim literature efficiently**.

### Why It Matters
- ⏱️ Saves researchers time during literature reviews
- 🧩 Enables automatic structure extraction for summarization
- 🧠 Forms a foundation for AI-assisted medical text understanding

---

## 📊 Project Overview

| Attribute | Details |
|-----------|---------|
| **Task** | Sequential sentence classification (5 classes) |
| **Dataset** | PubMed RCT 200k abstracts |
| **Classes** | Background • Objective • Methods • Results • Conclusions |
| **Model Accuracy** | **87%** |
| **Framework** | TensorFlow + Keras |
| **Interface** | Streamlit Web App |

---

## 🏗️ Model Architecture

The model combines **semantic**, **structural**, and **positional** features.

### 🧩 Inputs
- **Sentence text** → tokenized & embedded
- **Line number** → one-hot encoded (15 dims)
- **Total number of lines** → one-hot encoded (20 dims)

### ⚙️ Layers
1. **TextVectorization** → converts text to integer sequences
2. **Embedding (128-dim)** → learns dense word representations
3. **Conv1D + Global Max Pooling** → captures contextual patterns
4. **Dense layers** → process positional encodings
5. **Concatenate** token + position features
6. **Output layer (softmax)** → predicts one of 5 labels

```text
[Text] → [Vectorizer] → [Embedding + Conv1D] ┐
                                               ├─> Concatenate → Dense → Output (5 classes)
[Line number + Total lines] → [Dense layers] ┘
```

---

## 🚀 Streamlit App

A lightweight Streamlit UI lets users paste any research abstract and view predictions instantly.

### Run Locally

```bash
# Clone the repository
git clone https://github.com/St-Lexy/skimlit-reimagined.git
cd skimlit-reimagined

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Then open 👉 http://localhost:8501

### Example Prediction

| Sentence | Predicted Label |
|----------|----------------|
| Health is a fundamental pillar... | BACKGROUND |
| Using a user-centered design approach... | METHODS |
| Our evaluation... | CONCLUSIONS |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.87 |
| **Loss** | ~0.43 |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy + Label Smoothing (0.2) |

---

## 📁 Project Structure

```bash
skimlit-reimagined/
├── app.py                     # Streamlit app
├── model/                     # Saved model weights
├── data/                      # PubMed RCT dataset
└── README.md
```

---

## 🧠 Key Learnings

- Combining semantic and positional features drastically improves accuracy
- Conv1D layers efficiently capture local context within abstracts
- Label smoothing prevents overfitting and stabilizes training
- Streamlit enables fast ML deployment with minimal setup

---

## 🔮 Future Improvements

- [ ] Integrate BioBERT for domain-specific embeddings
- [ ] Add attention mechanisms for interpretability
- [ ] Enable live PubMed abstract classification via API
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Add confidence scores + visual explanations
- [ ] Support audio & video input for multimodal abstract analysis
- [ ] Add downloadable report of classified abstracts
- [ ] Support user feedback loops for model improvement

---

## 💡 Use Cases

| Audience | Benefit |
|----------|---------|
| **Researchers** | Rapidly locate abstract sections |
| **Healthcare Professionals** | Quickly find key outcomes |
| **ML Practitioners** | Learn hybrid feature engineering for NLP |

---

## 🛠️ Technologies

- **TensorFlow / Keras** — Model Training
- **Streamlit** — Web Interface
- **Scikit-learn** — OneHot Encoding + Metrics
- **Pandas & NumPy** — Data Handling
- **Matplotlib & Seaborn** — Visualization
- **Python 3.10+**

---

## 📚 References

- [SkimLit Paper](https://arxiv.org/abs/1710.06071) - Automating the classification of sentences in medical abstracts using NLP
- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👤 Author

**Amusan Olanrewaju Stephen (St. Lexy)**

🎓 Computer Science @ LAUTECH  
💻 Machine Learning & NLP Engineer  
📧 amusanolanrewaju420@gmail.com  
🌐 [st-lexy.github.io](https://st-lexy.github.io)  
🔗 [LinkedIn](https://www.linkedin.com/in/st-lexy) | [GitHub](https://github.com/St-Lexy)

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## ⭐ Support

If you find this project useful, don't forget to give it a star! ⭐

---

**Made with ❤️ by St. Lexy**
