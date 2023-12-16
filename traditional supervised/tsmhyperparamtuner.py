import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Loading the data
filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
data = pd.read_feather(filename)

X = data['review']
y = data['sentiment']

# Text preprocessing steps
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]  # Lowercasing and lemmatization
    return ' '.join(tokens)

X = X.apply(preprocess_text)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1],  # Regularization parameter
    'max_iter': [50, 100, 500, 1000],  # Maximum number of iterations
    'solver': ['liblinear', 'lbfgs'],  # Solvers
    'dual':[True, False], 
    'penalty':[None, 'l1', 'l2']
}

# Initialize Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Create GridSearchCV
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Perform grid search on training data
grid_search.fit(X_train_vec, y_train)

# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
