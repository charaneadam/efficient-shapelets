from sqlalchemy import create_engine
from peewee import PostgresqlDatabase
from peewee import Model

SAME_LENGTH_CANDIDATES_TABLE_NAME = "same_length_candidates"
VARIABLE_LENGTH_CANDIDATES_TABLE_NAME = "variable_length_candidates"

SAME_LENGTH_CLASSIFICATION_TABLE_NAME = "fixed_length_classification"
VARIABLE_CLASSIFICATION_LENGTH_TABLE_NAME = "variable_length_classification"

db_peewee = PostgresqlDatabase(
    database="shapelets", user="postgres", host="localhost", port=5432
)

engine = create_engine("postgresql://postgres:pass@localhost:5432/shapelets")
paper_engine = create_engine("postgresql://postgres:pass@localhost:5432/paper")


class BaseModel(Model):
    class Meta:
        database = db_peewee
