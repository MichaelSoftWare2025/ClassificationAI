import joblib

# Загрузка модели
model = joblib.load('ClassificationAI.pkl')

# Пример нового текста для классификации
new_text = ["Any text..."]

# Предсказание метки для нового текста
prediction = model.predict(new_text)
print(f"Label: {prediction[0]}")
