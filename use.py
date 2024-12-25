<<<<<<< HEAD
import joblib

# Загрузка модели
model = joblib.load('ClassificationAI.h5')

# Пример нового текста для классификации
new_text = ["Any text..."]

# Предсказание метки для нового текста
prediction = model.predict(new_text)
print(f"Label: {prediction[0]}")
=======
import joblib

# Загрузка модели
model = joblib.load('ClassificationAI.h5')

# Пример нового текста для классификации
new_text = ["Any text..."]

# Предсказание метки для нового текста
prediction = model.predict(new_text)
print(f"Label: {prediction[0]}")
>>>>>>> bcaec392123a85a16f632835d611e455d6128f93
