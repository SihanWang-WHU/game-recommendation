import csv
import redis
import psycopg2
import pymongo
from neo4j import GraphDatabase
from tqdm import tqdm
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
def import_data_to_postgres(csv_file_path, category, pg_conn, pg_cursor):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS games (
        app_id INT PRIMARY KEY,
        title VARCHAR(255),
        date_release DATE,
        win BOOLEAN,
        mac BOOLEAN,
        linux BOOLEAN,
        rating VARCHAR(50),
        positive_ratio INT,
        user_reviews INT,
        price_final DECIMAL(10, 2),
        price_original DECIMAL(10, 2),
        discount DECIMAL(5, 2),
        steam_deck BOOLEAN);
    
    CREATE TABLE IF NOT EXISTS recommendations (
        app_id INT,
        helpful INT,
        funny INT,
        date DATE,
        is_recommended BOOLEAN,
        hours DECIMAL(10, 1),
        user_id INT,
        review_id INT PRIMARY KEY,
        FOREIGN KEY (app_id) REFERENCES games(app_id));
    CREATE TABLE IF NOT EXISTS users (
        user_id INT PRIMARY KEY,
        products INT,
        reviews INT); """
    pg_cursor.execute(create_table_query)
    pg_conn.commit()

    if category == 'games':
        with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader):
                insert_query = """
                INSERT INTO games (app_id, title, date_release, win, mac, linux, 
                rating, positive_ratio, user_reviews, 
                price_final, price_original, discount, steam_deck)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = [row[field] for field in reader.fieldnames]
                pg_cursor.execute(insert_query, values)
            pg_conn.commit()

    if category == 'recommendations':
        with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader):
                insert_query = """
                INSERT INTO recommendations (app_id, helpful, funny, date, 
                is_recommended, hours, user_id, review_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = [row[field] for field in reader.fieldnames]
                pg_cursor.execute(insert_query, values)
            pg_conn.commit()

    if category == 'users':
        with open(csv_file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader):
                insert_query = """
                INSERT INTO users (user_id, products, reviews)
                VALUES (%s, %s, %s)
                """
                values = [row[field] for field in reader.fieldnames]
                pg_cursor.execute(insert_query, values)
            pg_conn.commit()

def import_data_to_neo4j(driver):
    with driver.session() as session:
        # 导入用户数据
        session.run("LOAD CSV WITH HEADERS FROM 'file:///users_DEMO.csv' AS line "
                    "CREATE (:User {userId: line.user_id, products: toInteger(line.products), reviews: toInteger(line.reviews)})")

        # 导入游戏数据
        session.run("LOAD CSV WITH HEADERS FROM 'file:///games.csv' AS line "
                    "CREATE (:Game {appId: line.app_id, title: line.title})")

        # 导入推荐关系
        session.run("LOAD CSV WITH HEADERS FROM 'file:///recommendations_DEMO.csv' AS line "
                    "MATCH (user:User {userId: line.user_id}) "
                    "MATCH (game:Game {appId: line.app_id}) "
                    "MERGE (user)-[:RECOMMENDS {hours: toFloat(line.hours), date: line.date, isRecommended: line.is_recommended = 'True'}]->(game)")


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
    users_fp = './dataset/users_DEMO.csv'
    recommendations_fp = './dataset/recommendations_DEMO.csv'
    games_fp = './dataset/games.csv'

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

    import_data_to_postgres(games_fp, 'games', pg_conn, pg_cursor)
    import_data_to_postgres(recommendations_fp, 'recommendations', pg_conn, pg_cursor)
    import_data_to_postgres(users_fp, 'users', pg_conn, pg_cursor)
    import_data_to_neo4j(neo4j_driver)

    # with neo4j_driver.session() as session:
    #     greeting = print_greeting(session, "hello, world")
    #     print(greeting)



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