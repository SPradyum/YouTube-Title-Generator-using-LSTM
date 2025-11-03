# =======================================
# 🚀 YouTube Title Generator (Optimized)
# Author: Pradyum S
# =======================================

import pandas as pd
import string
import numpy as np
import json
import tensorflow as tf
from keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
import keras.utils as ku
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
tf.random.set_seed(2)
np.random.seed(1)

# =======================================
# 1️⃣ Load Datasets
# =======================================
df1 = pd.read_csv(r'D:\Documents\Projects\Python\Title Generator\USvideos.csv')
df2 = pd.read_csv(r'D:\Documents\Projects\Python\Title Generator\CAvideos.csv')
df3 = pd.read_csv(r'D:\Documents\Projects\Python\Title Generator\GBvideos.csv')

data1 = json.load(open(r'D:\Documents\Projects\Python\Title Generator\US_category_id.json'))
data2 = json.load(open(r'D:\Documents\Projects\Python\Title Generator\CA_category_id.json'))
data3 = json.load(open(r'D:\Documents\Projects\Python\Title Generator\GB_category_id.json'))

def category_extractor(data):
    """Extract category id-title mapping."""
    ids = [int(item['id']) for item in data['items']]
    titles = [item['snippet']['title'] for item in data['items']]
    return dict(zip(ids, titles))

df1['category_title'] = df1['category_id'].map(category_extractor(data1))
df2['category_title'] = df2['category_id'].map(category_extractor(data2))
df3['category_title'] = df3['category_id'].map(category_extractor(data3))

df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates('video_id')

# =======================================
# 2️⃣ Filter "Entertainment" Titles
# =======================================
entertainment = df[df['category_title'] == 'Entertainment']['title'].tolist()

# =======================================
# 3️⃣ Text Cleaning
# =======================================
def clean_text(text):
    text = ''.join(ch for ch in text if ch not in string.punctuation).lower()
    return text.encode('utf8').decode('ascii', 'ignore')

corpus = [clean_text(t) for t in entertainment]

# =======================================
# 4️⃣ Tokenization and Sequence Building
# =======================================
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

# =======================================
# 5️⃣ Padding Sequences
# =======================================
def generate_padded_sequences(input_sequences):
    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(input_sequences)

# =======================================
# 6️⃣ Model Architecture (GRU)
# =======================================
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential([
        Embedding(total_words, 150, input_length=input_len),
        GRU(150, return_sequences=True),
        Dropout(0.2),
        GRU(100),
        Dense(total_words, activation='softmax')
    ])

    # Adam optimizer with learning rate decay
    optimizer = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = create_model(max_sequence_len, total_words)

# =======================================
# 7️⃣ Callbacks for Smarter Training
# =======================================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)

# =======================================
# 8️⃣ Model Training
# =======================================
history = model.fit(
    predictors, label,
    epochs=10,
    batch_size=512,
    validation_split=0.1,
    verbose=2,
    callbacks=[early_stop, lr_schedule]
)

# =======================================
# 9️⃣ Text Generation
# =======================================
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()

# =======================================
# 🔟 Example Prediction
# =======================================
print("\n🧠 Example Generated Title:")
print(generate_text("how to make", 5, model, max_sequence_len))
print(generate_text("Spiderman", 5, model, max_sequence_len))
print(generate_text("Dance", 5, model, max_sequence_len))
print(generate_text("Funny", 5, model, max_sequence_len))
