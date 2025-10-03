"""
Stock Sentiment Analysis
------------------------
This script uses NLP techniques (Bag of Words + RandomForestClassifier)
to classify financial news headlines as positive or negative sentiment
and predict stock market movement.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# Preprocessing function
def preprocess_data(data):
    data_cleaned = data.iloc[:, 2:27]
    data_cleaned.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    data_cleaned.columns = [str(i) for i in range(25)]
    for i in range(25):
        data_cleaned[str(i)] = data_cleaned[str(i)].str.lower()

    headlines = []
    for row in range(len(data_cleaned.index)):
        headlines.append(' '.join(str(x) for x in data_cleaned.iloc[row, 0:25]))
    return headlines


if __name__ == "__main__":
    file_path = "../data/Dataset.csv"
    dataframe = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Filter Apple-related news
    apple_related = dataframe.iloc[:, 2:27].apply(lambda x: x.str.contains('Apple', case=False, na=False))
    apple_data = dataframe[apple_related.any(axis=1)]

    # Split data
    train_data = apple_data[apple_data['Date'] < '20150101']
    test_data = apple_data[apple_data['Date'] > '20141231']

    # Preprocess train & test
    train_headlines = preprocess_data(train_data)
    test_headlines = preprocess_data(test_data)

    # Vectorization
    cv = CountVectorizer(ngram_range=(2, 2))
    train_data_modeled = cv.fit_transform(train_headlines)

    # Train model
    rc = RandomForestClassifier(n_estimators=200, criterion='entropy')
    rc.fit(train_data_modeled, train_data["Label"])

    # Prediction
    test_data_modeled = cv.transform(test_headlines)
    predictions = rc.predict(test_data_modeled)

    # Results
    predicted_movements = pd.DataFrame({
        'Date': test_data['Date'].reset_index(drop=True),
        'Prediction': predictions
    })

    predicted_movements['Predicted Movement'] = predicted_movements['Prediction'].map(
        {1: "Positive", 0: "Negative"}
    )

    print(predicted_movements[['Date', 'Predicted Movement']])
