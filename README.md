# 🐦 Twitter Sentiment Analysis

This project performs sentiment analysis on tweets and classifies them as **Positive**, **Negative**, or **Neutral** using a pre-trained RoBERTa model from Hugging Face.

## 🚀 Overview

Social media platforms contain a large amount of user opinions and feedback. This project analyzes the sentiment of tweets by combining text preprocessing with a powerful transformer-based language model.

The model processes a tweet and predicts the probability of each sentiment category.

## 🛠️ Technologies Used

* Python
* Transformers (Hugging Face)
* RoBERTa
* SciPy
* PyTorch

## ✨ Features

* Classifies tweets as Positive, Negative, or Neutral
* Handles user mentions and URLs during preprocessing
* Uses a pre-trained RoBERTa model trained on Twitter data
* Provides confidence scores for each sentiment class

## ⚙️ How It Works

1. Preprocess the tweet text
2. Replace usernames and links with standard tokens
3. Tokenize the text using the RoBERTa tokenizer
4. Generate sentiment predictions using the model
5. Apply Softmax to obtain sentiment probabilities

## ▶️ Installation

```bash
git clone https://github.com/krishnagll/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
pip install transformers torch scipy
```

Run the project:

```bash
python tw-sentiment.py
```

## 🧪 Example

**Input Tweet**

```text
This movie was amazing! I loved every moment of it.
```

**Output**

```text
Positive : 0.95
Neutral  : 0.04
Negative : 0.01
```

## 📚 Learning Outcomes

* Natural Language Processing (NLP)
* Transformer Models
* Sentiment Analysis
* Text Preprocessing
* Hugging Face Transformers

## 👨‍💻 Author

**Krishna Goyal**
Machine Learning Intern – Codec Technologies

GitHub: https://github.com/krishnagll
