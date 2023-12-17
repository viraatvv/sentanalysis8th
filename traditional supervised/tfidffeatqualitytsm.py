import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
data = pd.read_feather(filename)

X = data['review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000) 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = np.asarray(X_train_vec.mean(axis=0)).ravel()
top_scores = np.argsort(tfidf_scores)[-20:]

top_feature_words = [feature_names[i] for i in top_scores]
top_tfidf_scores = [tfidf_scores[i] for i in top_scores]

for word, score in zip(top_feature_words, top_tfidf_scores):
    print(f"Word: {word}, TF-IDF Score: {score}")
