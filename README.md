📧 Email Spam Detection Using Machine Learning
🧠 Overview

This project focuses on building a Machine Learning model to classify emails as spam or ham (non-spam) using Natural Language Processing (NLP) and supervised learning techniques.
The main objective is to improve email filtering accuracy by analyzing text features extracted from real-world email datasets.

🎯 Objectives

Analyze and preprocess email text data for noise removal.

Extract meaningful features using NLP techniques such as TF-IDF and Bag of Words (BoW).

Train and compare multiple classification algorithms for spam detection.

Evaluate model accuracy, precision, recall, and F1-score to identify the best-performing model.

📊 Dataset

Source: Public email spam datasets (e.g., SpamAssassin, UCI ML Repository).

Features: Email text, subject line, metadata (e.g., sender frequency, word frequency).

Target: Binary classification — Spam (1) / Ham (0).

⚙️ Technologies & Libraries

Programming Language: Python

Libraries Used:
pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, wordcloud

🔍 Methodology

Data Cleaning & Preprocessing:

Removed stopwords, punctuation, and special characters.

Tokenized and stemmed words using NLTK.

Feature Engineering:

Applied TF-IDF Vectorization and Count Vectorization (BoW).

Modeling:

Trained models using Naïve Bayes, Logistic Regression, SVM, and Random Forest.

Evaluation:

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

Visualization of classification performance and feature importance.

📈 Results

Multinomial Naïve Bayes achieved the best performance with 96% accuracy.

Model successfully generalized to unseen email samples with minimal false positives.

Visualized top indicative spam words (e.g., “free”, “offer”, “click”) using WordCloud.

💡 Key Insights

Text preprocessing and feature extraction significantly improve accuracy.

Lightweight models (like Naïve Bayes) perform better for sparse text data.

Combining NLP with ML enables robust, automated spam filtering for enterprise systems.
