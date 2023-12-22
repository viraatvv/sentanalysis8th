import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import time

filename = "https://github.com/lmassaron/datasets/releases/download/1.0/imdb_50k.feather"
reviews = pd.read_feather(filename)

X = reviews['review']

# Splitting the data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

latent_dims = [32, 64, 128, 256]  # Different latent dimensions to test
bce_train_losses = []
bce_test_losses = []
training_times = []

num_iterations = 10  # Number of iterations

for _ in range(num_iterations):
    bce_train_losses_iter = []
    bce_test_losses_iter = []
    training_times_iter = []

    for latent_dim in latent_dims:
        start_time = time.time()

        # Apply TruncatedSVD for dimensionality reduction on training and test data
        svd = TruncatedSVD(n_components=latent_dim, random_state=42)
        X_train_reduced = svd.fit_transform(X_train_vectorized)
        X_test_reduced = svd.transform(X_test_vectorized)

        # Reconstruct the data
        X_train_reconstructed = svd.inverse_transform(X_train_reduced)
        X_test_reconstructed = svd.inverse_transform(X_test_reduced)

        # Convert reconstructed matrices to dense if sparse
        if issparse(X_train_reconstructed):
            X_train_reconstructed = X_train_reconstructed.toarray()
            X_test_reconstructed = X_test_reconstructed.toarray()

        # Apply a clip to avoid logarithm of zero or very small values
        epsilon = 1e-10  # Small value to avoid division by zero
        X_train_reconstructed = np.clip(X_train_reconstructed, epsilon, 1 - epsilon)
        X_test_reconstructed = np.clip(X_test_reconstructed, epsilon, 1 - epsilon)

        # Calculate binary cross-entropy loss as an approximation for both training and testing data
        train_loss = -np.multiply(X_train_vectorized.data, np.log(X_train_reconstructed[X_train_vectorized.nonzero()] + epsilon)).sum()
        train_loss -= np.multiply(1 - X_train_vectorized.data, np.log(1 - X_train_reconstructed[X_train_vectorized.nonzero()] + epsilon)).sum()

        test_loss = -np.multiply(X_test_vectorized.data, np.log(X_test_reconstructed[X_test_vectorized.nonzero()] + epsilon)).sum()
        test_loss -= np.multiply(1 - X_test_vectorized.data, np.log(1 - X_test_reconstructed[X_test_vectorized.nonzero()] + epsilon)).sum()

        bce_train_losses_iter.append(train_loss / (X_train_vectorized.shape[0] * X_train_vectorized.shape[1]))
        bce_test_losses_iter.append(test_loss / (X_test_vectorized.shape[0] * X_test_vectorized.shape[1]))

        end_time = time.time()
        training_time = end_time - start_time
        training_times_iter.append(training_time)

    # Append results for this iteration to main lists
    bce_train_losses.append(bce_train_losses_iter)
    bce_test_losses.append(bce_test_losses_iter)
    training_times.append(training_times_iter)

# Convert lists to NumPy arrays for easier manipulation
bce_train_losses = np.array(bce_train_losses)
bce_test_losses = np.array(bce_test_losses)
training_times = np.array(training_times)

# Calculate means and standard deviations over iterations for losses and training times
mean_bce_train_losses = np.mean(bce_train_losses, axis=0)
std_bce_train_losses = np.std(bce_train_losses, axis=0)
mean_bce_test_losses = np.mean(bce_test_losses, axis=0)
std_bce_test_losses = np.std(bce_test_losses, axis=0)
mean_training_times = np.mean(training_times, axis=0)

# Plot binary cross-entropy losses for different latent dimensions
plt.figure(figsize=(8, 6))
plt.title('Mean Binary Cross-Entropy Losses for Different Latent Dimensions (averaged over 10 iterations)')
plt.errorbar(latent_dims, mean_bce_train_losses, yerr=std_bce_train_losses, fmt='o-', label='Training Loss')
plt.errorbar(latent_dims, mean_bce_test_losses, yerr=std_bce_test_losses, fmt='o-', label='Testing Loss')
plt.xlabel('Latent Dimension')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.show()

# Display mean training times for different latent dimensions
print("Mean Training Times for Different Latent Dimensions (averaged over 10 iterations):")
for i, dim in enumerate(latent_dims):
    print(f"Latent Dimension: {dim}, Mean Training Time: {mean_training_times[i]} seconds")

# Function for inference with user input
def predict_sentiment(user_input, svd_model, vectorizer):
    user_input_vectorized = vectorizer.transform([user_input])
    user_input_reduced = svd_model.transform(user_input_vectorized)
    user_input_reconstructed = svd_model.inverse_transform(user_input_reduced)

    epsilon = 1e-10
    reconstruction_error = -np.multiply(user_input_vectorized.data, np.log(user_input_reconstructed[user_input_vectorized.nonzero()] + epsilon)).sum()
    reconstruction_error -= np.multiply(1 - user_input_vectorized.data, np.log(1 - user_input_reconstructed[user_input_vectorized.nonzero()] + epsilon)).sum()

    return reconstruction_error / (user_input_vectorized.shape[0] * user_input_vectorized.shape[1])

# Getting user input for sentiment analysis
user_input_text = input("Enter your text for sentiment analysis: ")
result = predict_sentiment(user_input_text, svd, vectorizer)
print(f"Sentiment Analysis Score: {result}")