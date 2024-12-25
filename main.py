import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Шаг 1: Подготовка данных
# Предположим, что у вас есть набор данных с текстами и метками
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

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X = df['text']  # Тексты
y = df['label']  # Метки

# Разделяем данные на тренировочные и тестовые (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Создание модели с использованием Naive Bayes и CountVectorizer в пайплайне
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Шаг 4: Обучение модели
model.fit(X_train, y_train)

# Шаг 5: Вывод предсказанных меток и их соответствующих текстов на обучающих данных
print("Texts and Predicted Labels:")
for text, true_label, pred_label in zip(X_train, y_train, model.predict(X_train)):
    print(f"Text: {text}")
    print(f"Label: {true_label}, Предсказанная метка: {pred_label}\n")

joblib.dump(model, 'ClassificationAI.h5')