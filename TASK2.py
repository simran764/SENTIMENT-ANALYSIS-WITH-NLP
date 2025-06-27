import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# --- 1. Create a Sample Dataset of Customer Reviews ---
# In a real-world scenario, you would load this from a CSV, database, etc.
# '1' represents positive sentiment, '0' represents negative sentiment.
data = {
    'review': [
        "This product is amazing! I love it.",
        "Terrible experience, very disappointed.",
        "Great quality and fast delivery.",
        "Not good, don't recommend.",
        "Highly satisfied with my purchase.",
        "Worst customer service ever.",
        "Excellent value for money.",
        "Could be better, a bit flimsy.",
        "Fantastic product, worth every penny.",
        "Completely useless, waste of money.",
        "Very happy with the performance.",
        "Disappointing, expected more.",
        "A truly remarkable item.",
        "Awful, never buying again.",
        "Absolutely brilliant and effective."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print("Sample Dataset:")
print(df)
print("\n")

# --- 2. TF-IDF Vectorization ---
# Initialize the TF-IDF Vectorizer.
# max_features: Limits the number of features (words) to consider.
# stop_words='english': Removes common English stop words (e.g., 'the', 'is').
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit the vectorizer to the review text and transform the text into TF-IDF features.
X = tfidf_vectorizer.fit_transform(df['review'])
y = df['sentiment'] # Our target variable

print(f"TF-IDF Vectorization complete. Number of features (words): {X.shape[1]}\n")

# --- 3. Split Data into Training and Testing Sets ---
# 80% for training, 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")

# --- 4. Train the Logistic Regression Model ---
# Initialize the Logistic Regression model.
# random_state for reproducibility.
logistic_model = LogisticRegression(random_state=42)

# Train the model.
logistic_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.\n")

# --- 5. Evaluate the Model ---
# Make predictions on the test set.
y_pred = logistic_model.predict(X_test)

# Calculate accuracy.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on the test set: {accuracy:.2f}\n")

# Display classification report for more detailed metrics (precision, recall, f1-score).
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Display Confusion Matrix.
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- 6. Visualize Important Features (Word Coefficients) ---
# For logistic regression, the coefficients indicate the importance and direction
# (positive/negative) of each feature (word).

# Get feature names from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get the coefficients from the trained logistic regression model
# For binary classification, there's one set of coefficients.
coefficients = logistic_model.coef_[0]

# Create a DataFrame to store words and their coefficients
coeff_df = pd.DataFrame({'word': feature_names, 'coefficient': coefficients})

# Sort by coefficient to find the most positive and negative words
coeff_df = coeff_df.sort_values(by='coefficient', ascending=False)

# Get top N positive and negative words
top_n = 10 # Number of words to visualize
top_positive_words = coeff_df.head(top_n)
top_negative_words = coeff_df.tail(top_n).sort_values(by='coefficient', ascending=True)

# Concatenate for plotting
plot_data = pd.concat([top_negative_words, top_positive_words])

plt.figure(figsize=(12, 8))
sns.barplot(x='coefficient', y='word', data=plot_data, palette='coolwarm')
plt.title('Top 10 Most Influential Words for Sentiment (Logistic Regression Coefficients)', fontsize=16)
plt.xlabel('Coefficient Value (Positive indicates Positive Sentiment)', fontsize=12)
plt.ylabel('Word', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
file_path_vis = 'sentiment_analysis_word_coefficients.png'
plt.savefig(file_path_vis)
plt.close()

print(f"\nWord coefficients visualization saved to '{file_path_vis}'")
print("This plot shows which words are most associated with positive (right side) and negative (left side) sentiment.")
print("To view the visualization, open the 'sentiment_analysis_word_coefficients.png' file.")

# --- Optional: Predict on a new review ---
# new_review = ["This is an absolutely fantastic product, very happy!"]
# new_review_vectorized = tfidf_vectorizer.transform(new_review)
# prediction = logistic_model.predict(new_review_vectorized)
# sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
# print(f"\nPrediction for new review '{new_review[0]}': {sentiment_label}")
