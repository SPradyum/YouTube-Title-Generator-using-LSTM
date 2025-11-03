# ----------------------------------------------------------
# 📘 YouTube Title Generator using LSTM
# ----------------------------------------------------------
# This project trains an LSTM-based neural network on YouTube video titles
# (from multiple regions) to generate new titles in the "Entertainment" category.
# ----------------------------------------------------------

import pandas as pd
import string
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
tf.random.set_seed(2)
np.random.seed(1)

# ----------------------------------------------------------
# 🧾 STEP 1: Load datasets (YouTube Trending Video Data)
# ----------------------------------------------------------
df1 = pd.read_csv(r'D:\Documents\Projects\Python\Title Generator\USvideos.csv')
df2 = pd.read_csv(r'D:\Documents\Projects\Python\Title Generator\CAvideos.csv')
df3 = pd.read_csv(r'D:\Documents\Projects\Python\Title Generator\GBvideos.csv')

# ----------------------------------------------------------
# 📂 STEP 2: Load category mapping JSON files
# ----------------------------------------------------------
data1 = json.load(open(r'D:\Documents\Projects\Python\Title Generator\US_category_id.json'))
data2 = json.load(open(r'D:\Documents\Projects\Python\Title Generator\CA_category_id.json'))
data3 = json.load(open(r'D:\Documents\Projects\Python\Title Generator\GB_category_id.json'))

# ----------------------------------------------------------
# 🧩 STEP 3: Create a function to extract category names
# ----------------------------------------------------------
def category_extractor(data):
    ids = [int(item['id']) for item in data['items']]
    titles = [item['snippet']['title'] for item in data['items']]
    return dict(zip(ids, titles))

# ----------------------------------------------------------
# 🗂️ STEP 4: Map category titles to each dataframe
# ----------------------------------------------------------
df1['category_title'] = df1['category_id'].map(category_extractor(data1))
df2['category_title'] = df2['category_id'].map(category_extractor(data2))
df3['category_title'] = df3['category_id'].map(category_extractor(data3))

# ----------------------------------------------------------
# 🧮 STEP 5: Merge all dataframes and remove duplicates
# ----------------------------------------------------------
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.drop_duplicates('video_id')

# ----------------------------------------------------------
# 🎭 STEP 6: Filter titles under 'Entertainment' category
# ----------------------------------------------------------
entertainment_titles = df[df['category_title'] == 'Entertainment']['title']
entertainment_titles = entertainment_titles.tolist()

# ----------------------------------------------------------
# ✨ STEP 7: Clean the text (remove punctuation, lowercase)
# ----------------------------------------------------------
def clean_text(text):
    text = ''.join(e for e in text if e not in string.punctuation).lower()
    text = text.encode('utf8').decode('ascii', 'ignore')
    return text

corpus = [clean_text(title) for title in entertainment_titles]

# ----------------------------------------------------------
# 🔤 STEP 8: Tokenize text and create input sequences
# ----------------------------------------------------------
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words

input_sequences, total_words = get_sequence_of_tokens(corpus)

# ----------------------------------------------------------
# 🧱 STEP 9: Pad sequences to equal length
# ----------------------------------------------------------
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = to_categorical(label, num_classes=total_words)
    
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(input_sequences)

# ----------------------------------------------------------
# 🧠 STEP 10: Define and compile the LSTM model
# ----------------------------------------------------------
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Input embedding layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # LSTM layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(total_words, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model(max_sequence_len, total_words)

# ----------------------------------------------------------
# ⚙️ STEP 11: Train the model
# ----------------------------------------------------------
history = model.fit(predictors, label, epochs=20, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=3)])

# ----------------------------------------------------------
# 🧾 STEP 12: Generate new titles
# ----------------------------------------------------------
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()

# ----------------------------------------------------------
# 🧪 STEP 13: Example usage
# ----------------------------------------------------------
print("\n🎬 Example Generated Titles:\n")
print(generate_text("funny", 5, model, max_sequence_len))
print(generate_text("music", 5, model, max_sequence_len))
print(generate_text("best movie", 5, model, max_sequence_len))
print(generate_text("action", 5, model, max_sequence_len))
print(generate_text("spiderman", 5, model, max_sequence_len))