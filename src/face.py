import face_recognition
import numpy as np
import cv2
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Embedding

# Создаем подключение к базе данных
engine = create_engine('postgresql://postgres:postgres@localhost/faces')

# Создаем сессию
Session = sessionmaker(bind=engine)
session = Session()

# Объявление переменных
METHOD = "hog"  # алгоритм обнаружения лиц
font = cv2.FONT_HERSHEY_SIMPLEX  # шрифт для отображения имени
imgresize = 0.5  # коэффициент сжатия изображения
scale = 1 / imgresize  # коэффициент масштабирования
interval = 1  # интервал в секундах между распознаваниями
last_recognition_time = time.time()  # время последнего распознавания

# Указание программе с какой камеры считывать видеопоток
j = 0  # Номер подключенной камеры (может быть другим в вашем случае)
cap = cv2.VideoCapture(j)

# Настройка параметров камеры
width = 1280  # Ширина видеопотока
height = 720  # Высота видеопотока
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Переменные для хранения последнего распознавания
last_face_locations = []
last_face_names = []

while True:
    # Покадровое считывание изображения с камеры видеонаблюдения
    success, img = cap.read()

    # Трансформация изображения с помощью алгоритма resize
    imgS = cv2.resize(img, None, fx=imgresize, fy=imgresize, interpolation=cv2.INTER_AREA)

    # Изменение порядка цветовых каналов из BGR в RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Проверка времени для распознавания лиц
    current_time = time.time()
    if current_time - last_recognition_time >= interval:
        # Поиск лиц на текущем кадре
        face_locations = face_recognition.face_locations(imgS, model=METHOD)
        # Извлечение ключевых признаков лиц
        face_encodings = face_recognition.face_encodings(imgS, face_locations)

        # Обновление списка имен и координат лиц
        last_face_locations = []
        last_face_names = []

        # Цикл для сравнения лиц и определения принадлежности к базе данных
        for encodeFace, faceLoc in zip(face_encodings, face_locations):
            # Получение всех эмбеддингов из базы данных
            all_embeddings = session.query(Embedding).all()

            name = "Unknown"  # Изначально считаем, что лицо неизвестно
            if all_embeddings:
                # Вычисление расстояния между текущим эмбеддингом лица и эмбеддингами из базы данных
                face_distances = [np.linalg.norm(encodeFace - np.array(embedding.embedding)) for embedding in all_embeddings]
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= 0.6:
                    # Если нашлось совпадение, получаем соответствующее имя из базы данных
                    name = all_embeddings[best_match_index].person.last_name  # Фамилия найденного лица

            last_face_locations.append(faceLoc)
            last_face_names.append(name)

        # Обновление времени последнего распознавания
        last_recognition_time = current_time

    # Отображение рамок и имен лиц
    for (faceLoc, name) in zip(last_face_locations, last_face_names):
        # Масштабирование координат лица
        top, right, bottom, left = faceLoc
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)

        # Отображение лица и имени
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 3)

    # Отображение количества кадров в секунду
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), font, 1, (0, 255, 0), 2)

    # Отображение изображения
    cv2.imshow('Face Recognition', img)

    # Ожидание нажатия клавиши и проверка на Esc (код клавиши 27)
    k = cv2.waitKey(1)
    if k == 27:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
