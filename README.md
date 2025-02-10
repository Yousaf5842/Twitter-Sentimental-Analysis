#
This project focuses on performing sentiment analysis on textual data to classify sentiments into categories such as Positive, Negative, Neutral, and Suggestion. The goal is to preprocess raw text data, extract meaningful features, and train a deep learning model to achieve high accuracy in sentiment prediction.

Key Features:
Text preprocessing including cleaning, stop word removal, lemmatization, and vectorization.
Encoding sentiment labels into numerical categories for efficient model training.
Implementation of a deep learning model using TensorFlow and Keras for multi-class classification.
Evaluation using accuracy, confusion matrix, and visualization of predictions.
Ability to input custom text and predict sentiment in real-time.
Objectives
Develop a robust pipeline for cleaning and preprocessing text data.
Train a deep learning model to classify text into multiple sentiment categories.
Provide a user-friendly way to predict sentiment for custom inputs.
Methodology
Data Preprocessing:

Text cleaning: Remove special characters, numbers, and punctuation.
Lowercasing: Convert text to lowercase for consistency.
Stop word removal: Eliminate unnecessary common words.
Lemmatization: Reduce words to their root forms.
Vectorization: Convert cleaned text into numerical form using CountVectorizer.
Model Development:

Deep learning model with three dense layers (input, hidden, and output).
Optimizer: Adam with a custom learning rate.
Loss function: Sparse categorical crossentropy for multi-class classification.
Metrics: Accuracy to evaluate performance.
Model Evaluation:

Splitting data into training and test sets.
Evaluating accuracy on both training and validation data.
Confusion matrix to visualize classification performance.
Custom Prediction:

User input to predict sentiment using the trained model.
Results
Achieved an accuracy of over 95% on training data with optimized preprocessing and model design.
Effective sentiment classification across multiple categories.
Robust performance on unseen test data with room for further fine-tuning.
Future Scope
Improve model generalization on unseen data.
Explore advanced NLP techniques such as word embeddings (e.g., Word2Vec, GloVe).
Build a deployment-ready API or web application for real-time sentiment prediction.
