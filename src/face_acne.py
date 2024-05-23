import face_recognition
import numpy as np
import cv2
import torch
import time
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from torchvision.models import resnet18
import torch.nn as nn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Embedding, Person

# Создаем подключение к базе данных
engine = create_engine('postgresql://postgres:postgres@localhost/faces')

# Создаем сессию
Session = sessionmaker(bind=engine)
session = Session()

# объявление переменных
METHOD = "hog"  # алгоритм обнаружения лиц
font = cv2.FONT_HERSHEY_SIMPLEX  # шрифт для отображения имени
imgresize = 0.5  # коэффициент сжатия изображения
scale = 1 / imgresize  # коэффициент масштабирования
interval = 1  # интервал в секундах между распознаваниями
last_recognition_time = time.time()  # время последнего распознавания

# Загрузка модели для определения наличия болезни на лице
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device).eval()

model.load_state_dict(torch.load('src/etc/model_weights.pth', map_location='cpu'))
model.eval()

# Указание программе с какой камеры считывать видеопоток
j = 0  # Номер подключенной камеры (может быть другим в вашем случае)
cap = cv2.VideoCapture(j)

# Настройка параметров камеры
width = 1280  # Ширина видеопотока
height = 720  # Высота видеопотока
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Создание словаря для отслеживания положения и имени каждого лица
recognized_faces = {}

# Один раз выполните запрос для извлечения эмбеддингов и имен из базы данных
embeddings_and_names = session.query(Embedding.embedding, Person.last_name).join(Person).all()

# Преобразование полученных данных в словарь для удобства поиска по эмбеддингу
embeddings_dict = {tuple(embedding): name for embedding, name in embeddings_and_names}

while True:
    # Покадровое считывание изображения с камеры видеонаблюдения
    success, img = cap.read()

    # Проверка на успешное считывание изображения
    if not success:
        print("Не удалось считать изображение с камеры")
        break

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

        # Обновление информации о распознанных лицах
        recognized_faces = {}
        for encodeFace, faceLoc in zip(face_encodings, face_locations):
            name = "Unknown"  # Изначально считаем, что лицо неизвестно
            face_distances = [np.linalg.norm(encodeFace - np.array(db_embedding)) for db_embedding, name in embeddings_and_names]
            if face_distances:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= 0.6:
                    name = embeddings_and_names[best_match_index][1]  # Фамилия найденного лица

            # Масштабирование координат лица
            top, right, bottom, left = faceLoc
            top = int(top * scale)
            right = int(right * scale)
            bottom = int(bottom * scale)
            left = int(left * scale)

            recognized_faces[(top, right, bottom, left)] = name

        # Обновление времени последнего распознавания
        last_recognition_time = current_time

    # Отображение лиц и имён, сохранённых в recognized_faces
    for (top, right, bottom, left), name in recognized_faces.items():
        # Отображение лица и имени
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Получение и обработка лица моделью для классификации болезни
        face_image = Image.fromarray(img[top:bottom, left:right])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        face_image = transform(face_image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(face_image.to(device))
        probabilities = torch.softmax(outputs, dim=1)[0]
        class_names = ['Acne', 'Normal', 'Rosacea']  # Предположим, что это порядок классов

        # Вывод предсказаний на изображение
        text_x = right + 6
        text_y = top + 20
        for i, prob in enumerate(probabilities):
            class_name = class_names[i]
            text = f"{class_name}: {prob.item():.2f}"
            cv2.putText(img, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)
            text_y += 20

        # Отображение изображения
    cv2.imshow('Face Recognition', img)

    # Ожидание нажатия клавиши и проверка на Esc (код клавиши 27)
    k = cv2.waitKey(1)
    if k == 27:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
