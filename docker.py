import csv
import redis
import psycopg2
import pymongo
from neo4j import GraphDatabase
import random
import time
import json
import matplotlib.pyplot as plt
import numpy as np


# Function to import data to Redis
# def import_data_to_redis(csv_file_path, redis_conn):
#     with open(csv_file_path, 'r') as file:
#         reader = csv.DictReader(file)
#         # Iterate over each row in the CSV
#         for row in reader:
#             key = f"row:{row['Order ID']}"
#             redis_conn.hmset(key, row)
#             print(row)

# def import_data_to_redis(csv_file_path, redis_conn):
#
#
#
# def redis_query(redis_conn, query_type, query_value):
#
#
#
# # Function to connect to the database and import CSV data to PostgreSQL
# def import_data_to_postgres(csv_file_path, pg_conn, pg_cursor):
#
#
#
# # Function to execute query
# def query_launcher(query, redis_conn, pg_cursor, i=1):
#
#
#
# # Function to measure query performance
# def measure_performance(queries, redis_conn, pg_cursor):

# Example usage for Neo4j
def print_greeting(session, message):
    result = session.run("CREATE (a:Greeting) "
                         "SET a.message = $message "
                         "RETURN a.message + ', from node ' + id(a)", message=message)
    return result.single()[0]


# Main script execution
if __name__ == "__main__":
    # csv_file_path = 'ecommerce.csv'

    # Connect to Redis
    redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(
        dbname="postgresdb",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5439"
    )
    pg_cursor = pg_conn.cursor()

    # Connect to MongoDB
    mongo_username = 'mongouser'  # The username you set in docker-compose.yml
    mongo_password = 'mongopassword'  # The password you set in docker-compose.yml
    mongo_conn = pymongo.MongoClient(f"mongodb://{mongo_username}:{mongo_password}@localhost:27017/")
    mongo_db = mongo_conn["mydatabase"]

    # Connect to Neo4j
    neo4j_password = 'neopassword'  # The password you set in docker-compose.yml
    neo4j_uri = "bolt://localhost:7687"
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_password))

    with neo4j_driver.session() as session:
        greeting = print_greeting(session, "hello, world")
        print(greeting)

    # try:
    #     # Import data to Redis
    #     import_data_to_redis(csv_file_path, redis_conn)
    #
    #     # Import data to PostgreSQL
    #     import_data_to_postgres(csv_file_path, pg_conn, pg_cursor)
    #
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # finally:
    #     # Close the PostgreSQL cursor and connection
    #     if pg_cursor is not None:
    #         pg_cursor.close()
    #     if pg_conn is not None:
    #         pg_conn.close()
    #     # Redis connection does not need to be closed