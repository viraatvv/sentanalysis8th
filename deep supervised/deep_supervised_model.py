import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
reviews = pd.read_feather(filename)

X = reviews['review']
y = reviews['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

maxlen = 256
vocab_size = len(tokenizer.word_index) + 1

def tokenize_and_pad(texts, tokenizer, maxlen):
    seq = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    return padded_seq

X_train_padded = tokenize_and_pad(X_train, tokenizer, maxlen)
X_test_padded = tokenize_and_pad(X_test, tokenizer, maxlen)

model = Sequential()
voc = len(tokenizer.index_word) + 1
feats = 4
seq_len = 256
model.add(Embedding(voc, feats, input_length=seq_len))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

callback = EarlyStopping(patience=2)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
history = model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_test_padded, y_test), callbacks=[callback])
end_time = time.time() 

training_duration = end_time - start_time
print(f"Training duration: {training_duration} seconds")

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

def predict_sentiment(text, model, tokenizer, maxlen):
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequence, maxlen=maxlen)
    predicted_sentiment = model.predict(padded_sequence)[0][0]
    return predicted_sentiment

# Getting user input for sentiment analysis
user_input_text = input("Enter your text for sentiment analysis: ")

predicted_sentiment = predict_sentiment(user_input_text, model, tokenizer, maxlen)
if predicted_sentiment >= 0.5:
    print(f"The predicted sentiment score for '{user_input_text}' is positive: {predicted_sentiment}")
else:
    print(f"The predicted sentiment score for '{user_input_text}' is negative: {predicted_sentiment}")
