
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.datasets import imdb

# Define a function for text preprocessing
def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join([str(word) for word in text])
    return text.lower()

# Custom Logistic Regression class with L2 regularization
class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=3000, threshold=0.5, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.lambda_param = lambda_param  # Regularization parameter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y)) + (self.lambda_param / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return (predictions >= self.threshold).astype(int)


# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)

# Convert data to text
X_train_text = [preprocess_text(sample) for sample in X_train]
X_test_text = [preprocess_text(sample) for sample in X_test]

# Create development and test sets
X_train_dev, X_test_dev, y_train_dev, y_test_dev = train_test_split(X_train_text, y_train, test_size=0.2, random_state=42)

# Initialize CountVectorizer with the hyperparameters
m = 1000 #The maximum number of features to be used by the CountVectorizer
n = 0.9 #The high-frequency limit for words during the calculation of the CountVectorizer. It removes words that appear very frequently and do not provide much information.
k = 50 # The low-frequency limit for words during the calculation of the CountVectorizer.

vectorizer = CountVectorizer(max_features=m, max_df = n, min_df = k, binary=True)

# Fit and transform on the development set
X_train_count = vectorizer.fit_transform(X_train_dev).toarray()
X_test_count = vectorizer.transform(X_test_dev).toarray()

# Lists to store metrics
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []

training_sizes = [100, 500, 1000, 2000, 5000, 10000]

for size in training_sizes:
    logistic_regression = LogisticRegression(learning_rate=0.1, num_iterations=3000, threshold=0.5, lambda_param=0.01)
    logistic_regression.fit(X_train_count[:size], y_train_dev[:size])

    y_train_pred_lr = logistic_regression.predict(X_train_count[:size])
    y_test_pred_lr = logistic_regression.predict(X_test_count)

    train_accuracy = accuracy_score(y_train_dev[:size], y_train_pred_lr)
    test_accuracy = accuracy_score(y_test_dev, y_test_pred_lr)
    precision = precision_score(y_test_dev, y_test_pred_lr)
    recall = recall_score(y_test_dev, y_test_pred_lr)
    f1 = f1_score(y_test_dev, y_test_pred_lr)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


for size in training_sizes:
    start_time = time.time()  # Record start time
    logistic_regression = LogisticRegression(learning_rate=0.1, num_iterations=3000, threshold=0.5, lambda_param=0.01)
    logistic_regression.fit(X_train_count[:size], y_train_dev[:size])
    end_time = time.time()  # Record end time

    # Calculate runtime
    runtime = end_time - start_time
    print(f"Training size: {size}, Runtime: {runtime} seconds")

print(f"My Logistic Regression Train Accuracy: {train_accuracy:.5f}")
print(f"My Logistic Regression Test Accuracy: {test_accuracy:.5f}")

print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1 Score: {f1:.5f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test_dev, y_test_pred_lr))

# Plotting learning curves
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, train_accuracies, label='My Logistic Regression Training Accuracy')
plt.plot(training_sizes, test_accuracies, label='My Logistic Regression Test Accuracy')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Plotting performance metrics
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, precisions, label='Precision')
plt.plot(training_sizes, recalls, label='Recall')
plt.plot(training_sizes, f1_scores, label='F1 Score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.legend()
plt.show()