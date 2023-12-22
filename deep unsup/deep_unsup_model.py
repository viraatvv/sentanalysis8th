import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt

filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
reviews = pd.read_feather(filename)

X = reviews['review']

# Splitting the data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

max_words = 20000  # Maximum number of words to be considered
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

maxlen = 256
vocab_size = min(max_words, len(tokenizer.word_index) + 1)
embedding_dim = 32  # Reduced embedding dimension

def tokenize_and_pad(texts, tokenizer, maxlen):
    seq = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    return padded_seq

X_train_padded = tokenize_and_pad(X_train, tokenizer, maxlen)
X_test_padded = tokenize_and_pad(X_test, tokenizer, maxlen)

# Autoencoder architecture for unsupervised learning
input_seq = Input(shape=(maxlen,))
embedding = Embedding(vocab_size, embedding_dim)(input_seq)
encoder_lstm = LSTM(32)(embedding)

# Repeat the encoded vector maxlen times
repeat = RepeatVector(maxlen)(encoder_lstm)

# Decoder LSTM layer
decoder_lstm = LSTM(64, return_sequences=True)(repeat)
decoded = Dense(vocab_size, activation='softmax')(decoder_lstm)

autoencoder = Model(input_seq, decoded)
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  # Use sparse_categorical_crossentropy

callback = EarlyStopping(patience=2)

start_time = time.time()

# Training the autoencoder on input sequences
history = autoencoder.fit(X_train_padded, X_train_padded, epochs=10,
                          validation_data=(X_test_padded, X_test_padded), callbacks=[callback])

end_time = time.time()

training_duration = end_time - start_time
print(f"Training duration: {training_duration} seconds")


# Plotting training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function for inference with user input
def predict_sentiment(user_input, model, tokenizer, maxlen):
    user_input_seq = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_seq, maxlen=maxlen)
    predicted_output = model.predict(user_input_padded)
    return predicted_output

def calculate_sentiment_score(predicted_output):
    # Assuming the predicted_output contains probabilities for positive and negative sentiments
    avg_positive_prob = np.mean(predicted_output[:, :, :int(predicted_output.shape[2] / 2)])  # Average probabilities of positive sentiment
    avg_negative_prob = np.mean(predicted_output[:, :, int(predicted_output.shape[2] / 2):])  # Average probabilities of negative sentiment

    sentiment_score = avg_positive_prob - avg_negative_prob  # Calculate sentiment score
    return sentiment_score


# Getting user input for sentiment analysis
user_input_text = input("Enter your text for sentiment analysis: ")
predicted_output = predict_sentiment(user_input_text, autoencoder, tokenizer, maxlen)

print(f"Predicted Sentiment Score: {predicted_output}")

avg_sentiment_score = calculate_sentiment_score(predicted_output)
print(f"Average Sentiment Score: {avg_sentiment_score}")

