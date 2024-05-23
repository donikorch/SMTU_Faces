# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from db.models import Embedding, Person
# import face_recognition
# import cv2
#
# # Создаем подключение к базе данных
# engine = create_engine('postgresql://postgres:postgres@localhost/faces')
#
# # Создаем сессию
# Session = sessionmaker(bind=engine)
# session = Session()
#
# # Функция для обработки захваченного изображения и сохранения лица в базе данных
# def capture_and_save_face():
#     # Получение изображения с камеры
#     cap = cv2.VideoCapture(0)  # Укажите номер вашей камеры, если их несколько
#     ret, frame = cap.read()
#     cap.release()
#
#     # Обнаружение лиц на изображении
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     # Проверка наличия лица на изображении
#     if len(face_encodings) == 0:
#         print("Лицо не обнаружено.")
#         return
#
#     # Получение эмбеддинга лица
#     face_encoding = face_encodings[0]  # Предполагаем, что на изображении только одно лицо
#
#     # Запрос имени и фамилии у пользователя
#     first_name = input("Введите имя: ")
#     last_name = input("Введите фамилию: ")
#     middle_name = input("Введите Отчество: ")
#
#     # Сохранение эмбеддинга в базе данных
#     new_person = Person(first_name=first_name, last_name=last_name, middle_name=middle_name)
#     new_embedding = Embedding(embedding=face_encoding, person=new_person)
#     session.add(new_person)
#     session.add(new_embedding)
#     session.commit()
#
#     print("Лицо успешно сохранено в базе данных.")
#
# # Вызов функции для сохранения эмбеддинга в базе данных
# capture_and_save_face()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Embedding, Person
import face_recognition
import cv2

# Создаем подключение к базе данных
engine = create_engine('postgresql://postgres:postgres@localhost/faces')

# Создаем сессию
Session = sessionmaker(bind=engine)
session = Session()

def capture_and_save_face(image=None):
    if image is None:
        # Получение изображения с камеры
        cap = cv2.VideoCapture(0)  # Укажите номер вашей камеры, если их несколько
        ret, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(image)

    # Обнаружение лиц на изображении
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Проверка наличия лица на изображении
    if len(face_encodings) == 0:
        return False, None

    # Получение эмбеддинга лица
    face_encoding = face_encodings[0]  # Предполагаем, что на изображении только одно лицо
    return True, face_encoding

# Функция для сохранения лица в базе данных
def save_person_to_db(first_name, last_name, middle_name, face_encoding):
    new_person = Person(first_name=first_name, last_name=last_name, middle_name=middle_name)
    new_embedding = Embedding(embedding=face_encoding, person=new_person)
    session.add(new_person)
    session.add(new_embedding)
    session.commit()
    return True
