from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from models import Base  # Предполагается, что у вас есть базовый класс Base в models

# Создаем подключение к базе данных
engine = create_engine('postgresql://postgres:postgres@localhost/faces')

# Создаем функцию для сброса базы данных
def reset_database(engine):
    # Получаем метаданные
    meta = MetaData()

    # Отображаем текущие таблицы
    meta.reflect(bind=engine)

    # Дропаем все таблицы
    meta.drop_all(bind=engine)

    # Создаем таблицы заново
    Base.metadata.create_all(engine)

# Сбрасываем базу данных (если требуется)
reset_database(engine)

# Создаем таблицы, если их еще нет
Base.metadata.create_all(engine)

# Создаем сессию
Session = sessionmaker(bind=engine)
session = Session()