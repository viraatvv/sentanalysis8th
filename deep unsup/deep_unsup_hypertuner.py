import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Loading the data
filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
reviews = pd.read_feather(filename)

X = reviews['review']

# Splitting the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

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

# Define the Keras model function
def create_model(learning_rate=0.001, batch_size=32):
    model = Sequential()
    model.add(Input(shape=(maxlen,)))
    model.add(Embedding(vocab_size, 128, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(maxlen, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model

# Define a custom scorer for GridSearchCV (MSE as reconstruction error)
def custom_scorer(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Prepare the data (use X_train_padded as both input and target)
X_train_padded_target = X_train_padded  # Use X_train_padded as both input and target

# Create a wrapper function for the Keras model
def keras_model_wrapper(learning_rate=0.001, batch_size=32, epochs=5):
    model = create_model(learning_rate, batch_size)
    model.fit(X_train_padded_target, X_train_padded_target, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Create the model
model = keras_model_wrapper()

# Define hyperparameters for grid search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128]
}

# Create GridSearchCV instance
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring=custom_scorer)

# Fit GridSearchCV
grid_result = grid.fit(X_train_padded_target, X_train_padded_target)

# Print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    