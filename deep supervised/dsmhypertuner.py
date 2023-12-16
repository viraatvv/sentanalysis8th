import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
from kerastuner.tuners import RandomSearch

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

X_train_padded = tokenize_and_pad(X_train, tokenizer, maxlen=maxlen)
X_test_padded = tokenize_and_pad(X_test, tokenizer, maxlen=maxlen)

# Define a function that builds the model using hyperparameters
def build_model(hp):
    feats = hp.Int('feats', min_value=4, max_value=32, step=4)
    
    model = Sequential()
    model.add(Embedding(vocab_size, feats, input_length=maxlen))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set up the hyperparameter search
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='my_dir',
    project_name='my_project'
)

# Start the hyperparameter search
tuner.search(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test), callbacks=[EarlyStopping(patience=5)])


# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hp}")

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Training the best model
start_time = time.time()  # Start recording time
best_model.fit(X_train_padded, y_train, epochs=20, validation_data=(X_test_padded, y_test))
end_time = time.time()  # End recording time

# Calculating the training duration of the best model
training_duration = end_time - start_time
print(f"Training duration of the best model: {training_duration} seconds")

# Plotting accuracy and loss over time during training of the best model
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(best_model.history.history['accuracy'], label='Training Accuracy')
plt.plot(best_model.history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Time (Best Model)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_model.history.history['loss'], label='Training Loss')
plt.plot(best_model.history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time (Best Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
