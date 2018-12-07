from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

database_path = 'sqlite:///data/face_recognizer.db'
engine = create_engine(database_path)

Session = sessionmaker(bind=engine)
Base = declarative_base()


class Person(Base):
    __tablename__ = 'people'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    images = relationship('Image')


class Image(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True)
    path = Column(String)
    person_id = Column(Integer, ForeignKey('people.id'))
