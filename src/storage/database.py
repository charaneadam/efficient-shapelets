from peewee import PostgresqlDatabase
from peewee import Model

db = PostgresqlDatabase(
    database="shapelets", user="postgres", host="localhost", port=5432
)


class BaseModel(Model):
    class Meta:
        database = db
