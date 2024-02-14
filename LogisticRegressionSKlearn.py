
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.datasets import imdb

# Define a function for text preprocessing
def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join([str(word) for word in text])
    return text.lower()

# Load the IMDB movie reviews dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)

# Convert data to text
X_train_text = [preprocess_text(sample) for sample in X_train]
X_test_text = [preprocess_text(sample) for sample in X_test]

# Initialize CountVectorizer with the hyperparameters
m = 1000 #The maximum number of features to be used by the CountVectorizer
n = 0.9 #The high-frequency limit for words during the calculation of the CountVectorizer. It removes words that appear very frequently and do not provide much information.
k = 50 # The low-frequency limit for words during the calculation of the CountVectorizer.

# Convert data to Count vectors
vectorizer = CountVectorizer(binary=True, max_df=n, min_df=k, max_features=m)
X_train_Count = vectorizer.fit_transform(X_train_text)
X_test_Count = vectorizer.transform(X_test_text)

# Create development and test sets
X_train_dev, X_test_dev, y_train_dev, y_test_dev = train_test_split(X_train_text, y_train, test_size=0.2, random_state=42)

# Train the Logistic Regression Classifier on the development set
logistic_regression_classifier = LogisticRegression()
logistic_regression_classifier.fit(X_train_Count, y_train)

# Predictions on the development set
y_train_dev_pred_lr = logistic_regression_classifier.predict(X_train_Count)

# Predictions on the test set
y_test_dev_pred_lr = logistic_regression_classifier.predict(vectorizer.transform(X_test_dev))

# Results on the training set
print(f"Logistic Regression - Train Accuracy: {accuracy_score(y_train, y_train_dev_pred_lr):.5f}")
print(f"Logistic Regression - Test Accuracy: {accuracy_score(y_test_dev, y_test_dev_pred_lr):.5f}")

# Print the Classification Report
classification_report_str_lr = classification_report(y_test_dev, y_test_dev_pred_lr)
print("Logistic Regression - Classification Report:")
print(classification_report_str_lr)

# Lists to store metrics
train_accuracies_lr = []
test_accuracies_lr = []
precisions_lr = []
recalls_lr = []
f1_scores_lr = []

# Different training set sizes
training_sizes = [100, 500, 1000, 2000, 5000, 10000]

for size in training_sizes:
    # Train the Logistic Regression Classifier on a subset
    logistic_regression_classifier = LogisticRegression()
    logistic_regression_classifier.fit(X_train_Count[:size], y_train[:size])

    # Predictions on the training set
    y_train_pred_lr = logistic_regression_classifier.predict(X_train_Count[:size])

    # Predictions on the test set
    y_test_pred_lr = logistic_regression_classifier.predict(vectorizer.transform(X_test_dev))

    # Calculate metrics
    train_accuracy_lr = accuracy_score(y_train[:size], y_train_pred_lr)
    test_accuracy_lr = accuracy_score(y_test_dev, y_test_pred_lr)
    precision_lr = precision_score(y_test_dev, y_test_pred_lr)
    recall_lr = recall_score(y_test_dev, y_test_pred_lr)
    f1_lr = f1_score(y_test_dev, y_test_pred_lr)

    # Store metrics in the lists
    train_accuracies_lr.append(train_accuracy_lr)
    test_accuracies_lr.append(test_accuracy_lr)
    precisions_lr.append(precision_lr)
    recalls_lr.append(recall_lr)
    f1_scores_lr.append(f1_lr)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, train_accuracies_lr, label='Training Accuracy (Logistic Regression)')
plt.plot(training_sizes, test_accuracies_lr, label='Test Accuracy (Logistic Regression)')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves (Logistic Regression)')
plt.legend()
plt.show()

# Plot performance metrics
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, precisions_lr, label='Precision (Logistic Regression)')
plt.plot(training_sizes, recalls_lr, label='Recall (Logistic Regression)')
plt.plot(training_sizes, f1_scores_lr, label='F1 Score (Logistic Regression)')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Performance Metrics (Logistic Regression)')
plt.legend()
plt.show()
