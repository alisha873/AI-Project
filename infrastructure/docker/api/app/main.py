import json
import os

import psycopg2
import redis
from fastapi import FastAPI,Query


app = FastAPI(title="AutoShield API")


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "autoshield")
POSTGRES_USER = os.getenv("POSTGRES_USER", "autoshield")
POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
    "autoshield_dev_password",
)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")


def get_postgres_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=6379,
        decode_responses=True,
    )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/data")
def get_data():

    redis_client = get_redis_client()

    cached = redis_client.get("autoshield:data")

    if cached:
        return {
            "source": "redis",
            "data": json.loads(cached),
        }

    connection = get_postgres_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS application_data (
                    id SERIAL PRIMARY KEY,
                    message TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                SELECT id, message
                FROM application_data
                ORDER BY id
                """
            )

            rows = cursor.fetchall()
            connection.commit()

    finally:
        connection.close()

    data = [
        {
            "id": row[0],
            "message": row[1],
        }
        for row in rows
    ]

    redis_client.set(
        "autoshield:data",
        json.dumps(data),
        ex=60,
    )

    return {
        "source": "postgres",
        "data": data,
    }

@app.get("/search")
def search_data(message: str = Query(default="")):
    connection = get_postgres_connection()

    try:
        with connection.cursor() as cursor:
            query = (
                "SELECT id, message "
                "FROM application_data "
                f"WHERE message LIKE '%{message}%' "
                "ORDER BY id"
            )

            cursor.execute(query)
            rows = cursor.fetchall()

    finally:
        connection.close()

    return {
        "data": [
            {
                "id": row[0],
                "message": row[1],
            }
            for row in rows
        ]
    }