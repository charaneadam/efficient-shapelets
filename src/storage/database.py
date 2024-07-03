from sqlalchemy import create_engine
from peewee import PostgresqlDatabase
from peewee import Model

VARIABLE_LENGTH_CANDIDATES_TABLE_NAME = "random_lengths_candidates"
VARIABLE_LENGTH_TABLE_NAME = "classification_variable_lengths"
SAME_LENGTH_CLASSIFICATION_TABLE_NAME = "classification_same_lengths"
SAME_LENGTH_CANDIDATES_TABLE_NAME = "same_length_candidates"

db_peewee = PostgresqlDatabase(
    database="shapelets", user="postgres", host="localhost", port=5432
)

engine = create_engine("postgresql://postgres:pass@localhost:5432/shapelets")


class BaseModel(Model):
    class Meta:
        database = db_peewee
