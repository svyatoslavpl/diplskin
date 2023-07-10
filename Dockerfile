# Используйте базовый образ Python
FROM python:3.9

# Установка зависимостей проекта
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Копирование кода проекта в контейнер
COPY . /app

# Запуск тренировки модели при старте контейнера
CMD ["python", "train.py"]

