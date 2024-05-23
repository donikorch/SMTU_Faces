from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, ARRAY, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Создание подключения к базе данных
engine = create_engine('postgresql://postgres:postgres@localhost/faces')

# Создание базового класса
Base = declarative_base()


# Определение таблицы person
class Person(Base):
    __tablename__ = 'person'

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    middle_name = Column(String(255), nullable=False)
    embeddings = relationship('Embedding', back_populates='person')


# Определение таблицы embedding
class Embedding(Base):
    __tablename__ = 'embedding'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey('person.id'), nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    person = relationship('Person', back_populates='embeddings')


# Создание таблиц в базе данных
Base.metadata.create_all(engine)

# Создание сессии
Session = sessionmaker(bind=engine)
session = Session()