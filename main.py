# 🚀 YouTube Title Generator using LSTM (Optimized Version)

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ===========================
# 📂 LOAD & CLEAN DATA
# ===========================
data = pd.read_csv(r"D:\Documents\Projects\Python\Title Generator\youtube_titles.csv")

# Convert to lowercase and clean text
data['title'] = data['title'].astype(str).str.lower().str.replace(r'[^a-z\s]', '', regex=True)

titles = data['title'].tolist()

# ===========================
# 🔠 TOKENIZATION
# ===========================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(titles)
total_words = len(tokenizer.word_index) + 1

# Generate input sequences
input_sequences = []
for line in titles:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(2, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# ===========================
# 🧠 MODEL ARCHITECTURE
# ===========================
model = Sequential([
    Embedding(total_words, 150, input_length=max_sequence_len - 1),
    Bidirectional(LSTM(150, return_sequences=True)),
    Dropout(0.3),
    LSTM(100),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
              metrics=['accuracy'])

# ===========================
# ⏳ TRAINING CONFIGURATION
# ===========================
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=512,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

model.save("youtube_title_generator.keras")

# ===========================
# 🎯 TEXT GENERATION FUNCTION
# ===========================
def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Temperature sampling
        preds = np.log(predicted_probs + 1e-7) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        predicted = np.random.choice(len(preds), p=preds)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "":
            continue
        seed_text += " " + output_word
    return seed_text

# ===========================
# 🧪 TESTING THE MODEL
# ===========================
print("\n✨ Example Generated Titles:")
print(generate_text("how to make", 5, model, max_sequence_len, tokenizer, temperature=0.9))
print(generate_text("funny", 6, model, max_sequence_len, tokenizer, temperature=1.0))
print(generate_text("spiderman", 5, model, max_sequence_len, tokenizer, temperature=1.2))
print(generate_text("dance", 7, model, max_sequence_len, tokenizer, temperature=0.8))
