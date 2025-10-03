# 📈 Stock Price Prediction and Sentiment Analysis

## 🔹 Project Overview

This project aims to predict stock closing prices using **AI and Machine Learning** techniques, with a special focus on incorporating **sentiment analysis of financial news headlines** to enhance prediction accuracy.

By combining **time-series modeling (LSTM)** with **Natural Language Processing (NLP)** for sentiment, the project demonstrates how market psychology impacts stock movements.

---

## 🔹 Problem Statement

Accurate stock price prediction is a complex challenge due to market volatility and multiple influencing factors. This project focuses on:

* Building a robust LSTM model for stock price forecasting.
* Enhancing predictions with sentiment analysis of financial news data.
* Demonstrating how AI and ML can be applied in the finance domain.

---

## 🔹 Objectives

1. **Data Collection:**

   * Automated extraction of historical stock prices and financial data using APIs.
   * Libraries like `pandas_datareader` were used to fetch data.

2. **Preprocessing and Analysis:**

   * Cleaning and structuring the data.
   * Handling missing values, normalization, and feature selection.
   * Tools: **Pandas, NumPy, scikit-learn**.

3. **Model Development (Stock Price Prediction):**

   * Designed and trained **Long Short-Term Memory (LSTM)** models for time-series forecasting.

4. **Sentiment Analysis:**

   * Applied NLP techniques using **NLTK/spaCy**.
   * Performed sentiment classification on financial news headlines.
   * Integrated sentiment scores into the stock prediction pipeline.

---

## 🔹 Methodology

### 📊 Stock Prediction Using LSTM

1. **Data Preparation**

   * Loaded historical Apple stock trading data.
   * Normalized values for consistent scale.
   * Created sequences of 600-day windows for input and labels.

2. **Model Setup**

   * Configured LSTM layers with defined memory cells and input dimensions.
   * Implemented loss layers for training.

3. **Training Loop**

   * Trained LSTM on sequences with backpropagation.
   * Cleared input sequence after each iteration.

4. **Prediction**

   * Predicted next 1000 days of stock trading volumes.
   * De-normalized values for interpretation.

5. **Evaluation**

   * Metrics: **MSE, RMSE, MAE**.
   * Compared results with a naive forecast baseline.

6. **Visualization**

   * Plotted **actual vs predicted** trading volumes.
   * Displayed evaluation metrics.

---

### 📰 Stock Sentiment Analysis using NLP

1. **Data Preparation**

   * Collected Apple stock news headlines with sentiment labels.
   * Preprocessed text (cleaning, tokenization, lemmatization).

2. **Feature Representation**

   * Used embeddings (**Word2Vec, GloVe**) for vectorization.
   * Sequences padded to uniform length.

3. **Model Architecture**

   * Designed NLP model with recurrent/transformer layers.
   * Used dropout for regularization.

4. **Training**

   * Split dataset into train/test.
   * Evaluated with **accuracy, precision, recall, F1-score**.

5. **Prediction**

   * Applied trained model to unseen headlines for sentiment classification.

6. **Evaluation**

   * Visualized confusion matrices and ROC curves.
   * Analyzed false positives/negatives.

---

## 🔹 Results

* **Stock Price Prediction:** LSTM model successfully captured price patterns and outperformed baseline models.
* **Sentiment Analysis:** News sentiment demonstrated correlation with short-term stock movements.
* Visualizations include:

  * LSTM Output
    <img width="491" height="306" alt="LSTM Output" src="https://github.com/user-attachments/assets/e5c5ee6b-3de3-4a02-be50-84345a57187e" />

  * Sentiment Analysis Output
    <img width="875" height="263" alt="Sentiment Analysis Output" src="https://github.com/user-attachments/assets/ba247827-9239-4bb3-b26c-2d8a46df87e4" />


(*Add images from your `results/` folder here with `![](results/filename.png)`*)

---

## 🔹 Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, NLTK, spaCy, Gensim
* **Visualization:** Matplotlib, Seaborn
* **Deployment (optional):** Streamlit

---

## 🔹 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-sentiment-analysis.git
   cd stock-price-prediction-sentiment-analysis
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in `notebooks/` folder for step-by-step execution.

---

## 🔹 Future Work

* Integrate real-time stock and news feeds using APIs (Twitter, Bloomberg).
* Explore transformer-based models (e.g., BERT, FinBERT) for sentiment analysis.
* Build an interactive dashboard with Streamlit for live predictions.

---

## 🔹 Author

👤 **Aryan Warathe**

* 📧 Email: aryan08warathe@gmail.com
* 💼 LinkedIn: www.linkedin.com/in/aryan-warathe-75294598

---


