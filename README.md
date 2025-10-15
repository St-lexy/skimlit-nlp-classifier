# 📄 Skimlit: Sequential Sentence Classification for Medical Abstracts

An NLP-powered tool that automatically classifies sentences in medical research abstracts into their respective sections (Background, Objective, Methods, Results, Conclusions). This project uses Universal Sentence Encoder (USE) combined with Conv1D neural networks to achieve 80% classification accuracy on PubMed abstracts.

## 🎯 Problem Statement

Medical research papers contain critical information, but reading through lengthy abstracts to find specific sections is time-consuming. **Skimlit** automates the process of identifying which sentences belong to which section of a medical abstract, making literature review and research much faster.

### Why This Matters
- Researchers can quickly scan abstracts and jump to relevant sections
- Enables efficient literature review for systematic reviews and meta-analyses
- Improves accessibility of medical research
- Foundation for automated research summarization tools

## 📊 Project Overview

- **Task**: Sequential sentence classification (5 classes)
- **Dataset**: PubMed medical abstracts (~200K sentences)
- **Classes**: Background, Objective, Methods, Results, Conclusions
- **Model Architecture**: Universal Sentence Encoder + Conv1D
- **Accuracy**: 80%

## 🏗️ Model Architecture

### Feature Engineering Strategy

The model uses **three types of features** concatenated together:

1. **Token Embeddings** (USE - Universal Sentence Encoder)
   - 512-dimensional sentence embeddings
   - Captures semantic meaning of sentences
   - Pre-trained on large corpus for transfer learning

2. **Positional Features**
   - Line number (position in abstract)
   - Total number of lines in abstract
   - Helps model understand abstract structure

3. **Sequential Context**
   - Conv1D layers to capture patterns across sentences
   - Learns typical sentence ordering in medical abstracts

### Architecture Flow
```
Input Sentence
    ↓
Universal Sentence Encoder (512-dim embeddings)
    ↓
Concatenate with [Line Number, Total Lines]
    ↓
Conv1D Layers (pattern detection)
    ↓
Dense Layers
    ↓
Output (5 classes: Background, Objective, Methods, Results, Conclusions)
```

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **80%** |
| Baseline (Random) | 20% |
| Improvement vs Baseline | **4x better** |

### Why 80% is Strong Performance

Medical abstract classification is challenging because:
- Sentences can be ambiguous without full context
- Authors have varying writing styles
- Some sentences blend multiple categories
- Class boundaries are not always clear-cut

**Achieving 80% accuracy** means the model correctly identifies 4 out of 5 sentences, significantly speeding up literature review workflows.

## 🔧 Technical Implementation

### Key Features

**1. Universal Sentence Encoder (USE)**
- Pre-trained embedding model from TensorFlow Hub
- Captures semantic similarity better than traditional word embeddings
- Fixed 512-dimensional vectors for each sentence

**2. Positional Encoding**
- Line number: Where the sentence appears in the abstract (1, 2, 3...)
- Total lines: How many sentences in the entire abstract
- These features help the model learn structural patterns (e.g., Methods typically appear mid-abstract)

**3. Conv1D for Sequence Learning**
- Captures patterns across multiple sentences
- Learns that certain sentence types follow others
- Better than treating each sentence independently

### Data Preprocessing Pipeline
```python
1. Split abstracts into individual sentences
2. Label each sentence (Background/Objective/Methods/Results/Conclusions)
3. Generate USE embeddings (512-dim vectors)
4. Extract positional features (line_number, total_lines)
5. Concatenate all features
6. Feed into Conv1D + Dense network
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
TensorFlow Hub
NumPy
Pandas
Scikit-learn
```

### Installation
```bash
# Clone the repository
git clone https://github.com/St-Lexy/skimlit-nlp-classifier.git
cd skimlit-nlp-classifier

# Install dependencies
pip install -r requirements.txt
```

### Dataset
- **Source**: PubMed 200k RCT dataset
- **Format**: Medical abstracts with sentence-level labels
- **Download**: Available through TensorFlow Datasets or PubMed directly

### Training the Model
```bash
# Open and run the Jupyter notebook
jupyter notebook skimlit_classifier.ipynb
```

### Making Predictions
```python
# Example: Classify sentences in a new abstract
abstract = [
    "Diabetes is a major health concern worldwide.",
    "This study aims to evaluate the efficacy of a new treatment.",
    "We conducted a randomized controlled trial with 500 participants.",
    "The treatment showed a 30% improvement in glucose control.",
    "Our findings suggest this treatment is effective for Type 2 diabetes."
]

predictions = model.predict(abstract)
# Output: ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
```

## 📁 Project Structure
```
skimlit-nlp-classifier/
├── README.md
├── requirements.txt
├── skimlit_classifier.ipynb     # Main training notebook
├── data/                         # Dataset directory
│   └── pubmed_abstracts/
├── models/                       # Saved model weights
│   └── skimlit_model.h5
├── results/                      # Performance metrics
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── sample_predictions.png
└── src/                         # Utility scripts (optional)
    ├── preprocess.py
    └── evaluate.py
```

## 🎓 Key Learnings

1. **Feature Engineering is Crucial**: Combining semantic embeddings with positional features significantly improved performance over embeddings alone.

2. **Transfer Learning Works for NLP**: Universal Sentence Encoder's pre-trained weights provided strong baseline performance without training from scratch.

3. **Context Matters**: Conv1D layers helped capture sequential patterns (e.g., Methods sections typically follow Objectives).

4. **Domain-Specific Challenges**: Medical language is specialized, but transfer learning from general text still provides good foundation.

## 🔮 Future Improvements

- [ ] Experiment with BERT/BioBERT for better medical domain understanding
- [ ] Add attention mechanisms to focus on key phrases
- [ ] Implement bidirectional context (look at surrounding sentences)
- [ ] Try LSTM/GRU for better sequence modeling
- [ ] Ensemble multiple models for improved accuracy
- [ ] Deploy as web API for real-time classification
- [ ] Extend to full-text papers (not just abstracts)
- [ ] Add confidence scores for predictions

## 💡 Use Cases

**For Researchers:**
- Quickly scan abstracts to find methodology details
- Extract results sections for systematic reviews
- Organize literature by abstract structure

**For Healthcare Professionals:**
- Rapidly assess clinical trial designs
- Find key outcomes without reading full abstracts
- Efficient evidence-based medicine practice

**For ML Engineers:**
- Example of combining multiple feature types
- Template for sequential NLP classification
- Demonstration of transfer learning in NLP

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x / Keras** - Deep learning framework
- **TensorFlow Hub** - Universal Sentence Encoder
- **Conv1D** - Sequence pattern detection
- **NumPy & Pandas** - Data manipulation
- **Scikit-learn** - Metrics and evaluation
- **Matplotlib** - Visualization

## 📚 Dataset Information

**PubMed 200k RCT Dataset**
- ~200,000 sentences from medical abstracts
- Randomized Controlled Trial (RCT) papers
- Sentence-level labels for 5 categories
- Publicly available for research purposes

## 🔍 Model Interpretation

The model learns:
- **Background sentences** often contain general context and motivation
- **Objective sentences** state research goals and hypotheses
- **Methods sentences** describe experimental procedures and protocols
- **Results sentences** present findings and statistical outcomes
- **Conclusions sentences** summarize implications and future work

## 👤 Author

**Amusan Olanrewaju Stephen**
- GitHub: [@St-Lexy](https://github.com/St-Lexy)
- LinkedIn: [olanrewaju-amusan](https://linkedin.com/in/olanrewaju-amusan)
- Email: amusanolanrewaju420@gmail.com
- Portfolio: [st-lexy.github.io](https://st-lexy.github.io)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- PubMed for providing the dataset
- TensorFlow Hub for Universal Sentence Encoder
- The NLP research community for inspiration

---

⭐ If you found this project helpful for your research or work, please consider giving it a star!

## 📖 Related Reading

- [Original Skimlit Paper (2017)](https://arxiv.org/abs/1710.06071)
- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- [Sequential Sentence Classification in Medical Abstracts](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017717/)
