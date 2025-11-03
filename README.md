# 🧠 LSTM YouTube Title Generator

This project demonstrates how an **LSTM neural network** can learn patterns in YouTube video titles and generate new, creative titles.  
Using **TensorFlow (Keras)**, **Pandas**, and **JSON datasets**, the model processes text data from real-world YouTube datasets and predicts the next possible word in a sequence.

---

## 🚀 Features
- Combines and cleans data from **multiple country datasets (US, CA, GB)**
- Extracts and maps **YouTube video categories** using JSON files
- Preprocesses and tokenizes text titles
- Builds and trains an **LSTM-based model** with Keras
- Uses **categorical cross-entropy loss** and **Adam optimizer**
- Generates new titles based on user-provided seed text

---

## 🧩 Technologies Used
- **Python 3.10+**
- **TensorFlow / Keras**
- **Pandas**
- **NumPy**
- **JSON**
- **NLTK (for tokenization, optional)**

---
⚙️ How to Run
- Download the required datasets:
- USvideos.csv, CAvideos.csv, GBvideos.csv
- US_category_id.json, CA_category_id.json, GB_category_id.json
- **Update the dataset file paths in the Python script.**
---
After training, you can generate new titles:

- **_print(generate_text("My new", 5, model, max_sequence_len))_**
