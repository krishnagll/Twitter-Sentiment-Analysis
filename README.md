# Twitter Sentiment Analysis using RoBERTa and NLTK

This repository contains a state-of-the-art Sentiment Analysis tool developed as part of my Machine Learning Internship at Codec Technologies. The project classifies Twitter data (tweets) into three distinct categories: **Positive**, **Negative**, or **Neutral**.

## 🚀 Overview
Instead of relying solely on traditional lexicon-based methods, this project utilizes a hybrid approach:
1. **NLTK (Natural Language Toolkit):** Used for robust social-media tokenization via `TweetTokenizer` to cleanly handle handles, URLs, and punctuation.
2. **Hugging Face Transformers (RoBERTa):** Utilizes the deep-learning model `cardiffnlp/twitter-roberta-base-sentiment`, which is specifically pre-trained on millions of tweets for highly accurate contextual understanding.

## 🛠️ Features
- **Advanced Preprocessing:** Automatically sanitizes and normalizes user mentions (`@user`) and links (`http`).
- **Emoji-Aware Tokenization:** Retains emotional markers like emojis intact during splitting.
- **Deep Learning Inference:** Outputs probability distributions across negative, neutral, and positive labels using Softmax activation.

## 📦 Installation & Setup

To run this project locally, clone the repository and install the required dependencies:

```bash
git clone [https://github.com/krishnagll/Twitter-Sentiment-Analysis.git](https://github.com/krishnagll/Twitter-Sentiment-Analysis.git)
cd Twitter-Sentiment-Analysis
pip install transformers torch scipy nltk
