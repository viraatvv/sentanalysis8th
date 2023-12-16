import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# Loading the data
filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
reviews = pd.read_feather(filename)

X = reviews['review']
y = reviews['sentiment']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Tokenizing and padding sequences
maxlen = 256
vocab_size = len(tokenizer.word_index) + 1

def tokenize_and_pad(texts, tokenizer, maxlen):
    seq = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    return padded_seq

X_train_padded = tokenize_and_pad(X_train, tokenizer, maxlen)
X_test_padded = tokenize_and_pad(X_test, tokenizer, maxlen)

# Building the model
model = Sequential()
voc = len(tokenizer.index_word) + 1
feats = 8
seq_len = 256
model.add(Embedding(voc, feats, input_length=seq_len))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

callback = EarlyStopping(patience=5)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model and recording time
start_time = time.time()  # Start recording time
history = model.fit(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test), callbacks=[callback])
end_time = time.time()  # End recording time

# Calculating the training duration
training_duration = end_time - start_time
print(f"Training duration: {training_duration} seconds")

# Plotting accuracy and loss over time during training
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
