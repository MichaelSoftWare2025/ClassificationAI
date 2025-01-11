import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import argparse

data = {
    'text': [
        'I love this movie', 'This movie is terrible', 'I enjoyed the book', 
        'The movie was boring', 'Fantastic performance!', 'Worst film ever', 
        'I will watch it again', 'It was okay', 'Really bad experience', 'Amazing movie',
        'I love programming!', "I don't love programming", "I don't like programming", "I like programming!", "I love count!"
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'neutral', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive', 'positive']
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

model.score(X_test, y_test)

argparser = argparse.ArgumentParser()

argparser.add_argument('--text', type=str, help='Text to classify')
argparser.add_argument('--build', action='store_true', help='Build the model')
args = argparser.parse_args()

text = args.text

if text:
    prediction = model.predict([text])
    print(f"Label: {prediction[0]}")
else:
    if args.build:
        print("Building the model...")
        joblib.dump(model, 'ClassificationAI.pkl')
    else:
        print("Please provide a text to classify or use the --build option to train the model.")