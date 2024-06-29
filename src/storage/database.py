from sqlalchemy import create_engine
from peewee import PostgresqlDatabase
from peewee import Model

db_peewee = PostgresqlDatabase(
    database="shapelets", user="postgres", host="localhost", port=5432
)

engine = create_engine("postgresql://postgres:pass@localhost:5432/shapelets")


class BaseModel(Model):
    class Meta:
        database = db_peewee
