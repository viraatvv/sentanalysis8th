import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from time import time
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
data = pd.read_feather(filename)

X = data['review']
y = data['sentiment']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text) 
    tokens = word_tokenize(text)  
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

X = X.apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=50, solver='lbfgs', 
                           penalty=None, C=0.001, dual=False)

iterations = []
train_losses = []
test_losses = []
train_accuracies = []

start = time()

for i in range(1, 11):  
    model.fit(X_train_vec, y_train)
    
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)
    
    train_accuracy = (y_train_pred == y_train).mean()
    test_accuracy = (y_test_pred == y_test).mean()
    
    epsilon = 1e-15
    prob_train = np.clip(model.predict_proba(X_train_vec)[:, 1], epsilon, 1 - epsilon)
    prob_test = np.clip(model.predict_proba(X_test_vec)[:, 1], epsilon, 1 - epsilon)
    
    train_loss = -((y_train * np.log(prob_train) + (1 - y_train) * np.log(1 - prob_train))).mean()
    test_loss = -((y_test * np.log(prob_test) + (1 - y_test) * np.log(1 - prob_test))).mean()
    
    iterations.append(i)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    
    print(f"Iteration {i}: Train Loss - {train_loss:.4f}, Train Acc - {train_accuracy:.4f}, Test Loss - {test_loss:.4f}, Test Acc - {test_accuracy:.4f}")
    
end = time()

print(f'Training Time: {end - start}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(iterations, train_losses, label='Train Loss')
plt.plot(iterations, test_losses, label='Test Loss')
plt.title('Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(iterations, train_accuracies, label='Train Accuracy')
plt.title('Training Accuracy Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
