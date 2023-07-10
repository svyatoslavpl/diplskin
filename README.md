# ## Проект обучения модели распознавания кожных заболеваний

Проект представляет собой модель машинного обучения для распознавания кожных заболеваний на основе набора данных HAM10000. В проекте доступны два варианта обучения модели: обычное обучение и обучение с использованием механизма Soft Attention.

### Структура проекта

- **data**: Каталог, содержащий данные для обучения и тестирования модели.
    - **HAM10000**: Каталог, содержащий изображения для обучения и тестирования.
        - **train_dir**: Каталог с изображениями для обучения модели.
        - **test_dir**: Каталог с изображениями для тестирования модели.
    - **HAM10000_metadata.csv**: CSV-файл, содержащий метаданные для набора данных HAM10000.

- **models**: Каталог, содержащий модуль `soft_attention.py`, реализующий механизм Soft Attention для модели.

- **utils**: Каталог, содержащий утилитарные модули для обработки данных и метрик модели.
    - **data_processing.py**: Модуль для обработки данных и создания тренировочных и тестовых наборов.
    - **metrics.py**: Модуль с определенными метриками для оценки модели.

- **train.py**: Основной скрипт для обучения модели с использованием обычного подхода.

- **evaluate.py**: Скрипт для оценки обученной модели на тестовых данных.

- **train_without_SA.py**: Скрипт для обучения модели без использования механизма Soft Attention.

### Как использовать

1. Убедитесь, что у вас установлен Docker и Docker Compose.

2. Склонируйте репозиторий проекта:

   ```bash
   git clone https://github.com/your_username/your_project.git
   ```

3. Перейдите в каталог проекта:

   ```bash
   cd your_project
   ```

4. Соберите Docker-контейнер:

   ```bash
   docker-compose build
   ```

5. Запустите обучение модели с использованием обычного подхода:

   ```bash
   docker-compose run app python train.py
   ```

   Модель будет обучена на данных из каталога `data/HAM10000/train_dir`. Веса модели будут сохранены в каталог `models/ResNet50.h5`.

6. Запустите обучение модели с использованием механизма Soft Attention:

   ```bash
   docker-compose run app python train.py
   ```

   Модель будет обучена с использованием механизма Soft Attention на данных из каталога `data/HAM10000/train_dir`. Веса модели с механизмом Soft Attention будут сохранены в каталог `models/ResNet50+SA.h5`.

7. Оцените обученную модель на тестовых данных:

   ```bash
   docker-compose run app python evaluate.py
   ```

   Скрипт оценит обученную модель на данных из каталога `data/HAM10000/test_dir` и выведет метрики оценки.

### Замечания

- Модель обучается на наборе данных HAM10000 для распознавания кожных заболеваний.

- Если вы хотите использовать собственные данные для обучения, замените соответствующие файлы и каталоги внутри каталога `data`.

- Для изменения гиперпараметров обучения или других настроек, внесите соответствующие изменения в код скриптов `train.py` и `train_without_SA.py`.

- При необходимости можете изменить структуру проекта или добавить новые функции в утилитарные модули в каталоге `utils`.

Теперь вы можете легко обучать модель одной командой и сохранять веса модели в директории проекта. Также есть возможность выбрать вариант обучения модели как обычной, так и с использованием механизма Soft Attention.
