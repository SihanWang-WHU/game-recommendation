import csv
import redis
import psycopg2
import random
import time
from datetime import datetime 
import json
import re
import pandas as pd
import pymongo
from neo4j import GraphDatabase
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

postgres_db_name = 'postgresdb'
postgres_password = 'postgres'
postgres_user = 'postgres'
postgres_port = '5439'
neo4j_username = 'neo4j'
neo4j_password = 'neopassword'
mongo_username = 'mongouser'
mongo_password = 'mongopassword'

# csv_file_path_all = ['/Users/chiuchiu/Desktop/game-recommendation-main/dataset/games.csv', '/Users/chiuchiu/Desktop/game-recommendation-main/dataset/recommendations.csv', '/Users/chiuchiu/Desktop/game-recommendation-main/dataset/users.csv']
csv_file_path_all = ['./dataset/games.csv', './dataset/recommendations.csv', './dataset/users.csv']


def infer_sql_data_type(dtype, max_len=None):
    # Map pandas dtypes to SQL data types
    if "int" in str(dtype):
        return "INT"
    # json里面不能用decimal
    elif "float" in str(dtype):
        return "FLOAT"
    elif "datetime" in str(dtype):
        return "DATETIME"
    elif "bool" in str(dtype):
        return "BOOLEAN"
    elif "object" in str(dtype): # object dtype in pandas is typically a string
        if max_len==1:
            return "BOOLEAN"
        else:
            return f"VARCHAR({int(max_len)})"
    else:
        return "VARCHAR(255)"  # Default


def generate_create_table_query(csv_file, table_name):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Construct CREATE TABLE query
    sql_fields = []
    for col in df.columns:
        # Handle special characters and reserved SQL words
        col_escaped = f"{col}"
        
        # Compute max length for string columns
        max_len = None
        if df[col].dtype == 'object':
            max_len = df[col].str.len().max()

        sql_dtype = infer_sql_data_type(df[col].dtype, max_len)
        sql_fields.append(f"{col_escaped} {sql_dtype}")

    fields_str = ", ".join(sql_fields)
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({fields_str});"
    return query


# Function to connect to the database and import CSV data to PostgreSQL
def import_data_to_postgres(csv_file_path, create_table_sql, pg_conn, pg_cursor):

    # Execute the SQL statement to create the table
    # pg_cursor.execute("SET search_path TO test")
    # pg_conn.commit()
    pg_cursor.execute(create_table_sql)
    pg_conn.commit()
    f=pd.read_csv(csv_file_path)

    f_sql = f.astype(object).where(pd.notnull(f), None)  
    values = f_sql.values.tolist()  
    s = ','.join(['%s' for _ in range(len(f.columns))])
    insert_sql = 'INSERT INTO {} ({}) VALUES ({})'.format(
    csv_file_path.split('/')[-1].split('.')[0],
    ', '.join(f.columns),  # Join the column names with commas
    ', '.join(['%s'] * len(f.columns))  # Use placeholders for values (assuming you're using parameterized queries)
    )
    pg_cursor.executemany(insert_sql, values)
    pg_conn.commit()


# Define your SQL queries as strings
queries = [
    "select title from games where mac=FALSE;",
    "select g.app_id, g.title from games as g join recommendations r on g.app_id = r.app_id where r.helpful > 0",
    "select u.user_id from users as u join recommendations r on u.user_id = r.user_id where u.products > 10",
    "select g.title as recommended_games from"
    "(select r.app_id from users as u join recommendations r on u.user_id = r.user_id where u.products > 5 and u.reviews > 1 and r.is_recommended = TRUE) as hot_users "
    "join games g on g.app_id = hot_users.app_id "
]

def query_launcher(query, redis_conn, pg_cursor):
    redis_key = 'query_result:' + query
    cached_result = redis_conn.get(redis_key)
    
    if cached_result is not None:
        print(f"Cache hit for query: {query}")
        return json.loads(cached_result), 'cache'
    else:
        # If there is no cached result, execute the query on PostgreSQL
        pg_cursor.execute(query)
        result = pg_cursor.fetchall()
        
        # Cache the new result in Redis, serializing the result as a JSON string
        redis_conn.setex(redis_key, 3600, json.dumps(result))
        print(f"Cache miss for query: {query}")
        return result, 'db'


def import_data_to_neo4j(driver):
    with driver.session() as session:
        # 导入用户数据
        session.run("LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS line "
                    "CREATE (:User {userId: line.user_id, products: toInteger(line.products), reviews: toInteger(line.reviews)})")
        print('users file imported')

        # 导入游戏数据
        session.run("LOAD CSV WITH HEADERS FROM 'file:///games.csv' AS line "
                    "CREATE (:Game {appId: line.app_id, title: line.title})")
        print('games file imported')

        # 导入推荐关系
        session.run("LOAD CSV WITH HEADERS FROM 'file:///recommendations.csv' AS line "
                    "MATCH (user:User {userId: line.user_id}) "
                    "MATCH (game:Game {appId: line.app_id}) "
                    "MERGE (user)-[:RECOMMENDS {hours: toFloat(line.hours), date: line.date, isRecommended: line.is_recommended = 'True'}]->(game)")
        print('recommendations file matched and imported')

        # # Import User Data
        # session.run("LOAD CSV WITH HEADERS FROM 'file:///var/lib/neo4j/import/users.csv' AS line "
        #             "CREATE (:User {userId: line.user_id, products: toInteger(line.products), reviews: toInteger(line.reviews)})")
        # # Import Game Data
        # session.run("LOAD CSV WITH HEADERS FROM 'file:///var/lib/neo4j/import/games.csv' AS line "
        #             "CREATE (:Game {appId: line.app_id, title: line.title})")
        # # Import Recommendation Relationships
        # session.run("LOAD CSV WITH HEADERS FROM 'file:///var/lib/neo4j/import/recommendations.csv' AS line "
        #             "MATCH (user:User {userId: line.user_id}) "
        #             "MATCH (game:Game {appId: line.app_id}) "
        #             "MERGE (user)-[:RECOMMENDS {hours: toFloat(line.hours), date: line.date, isRecommended: line.is_recommended = 'True'}]->(game)")



# Main script execution
if __name__ == "__main__":

    ### IMPORT DATA TO DIFFERENT DATABASES
    redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

    pg_conn = psycopg2.connect(
        dbname=postgres_db_name,
        user=postgres_user,
        password=postgres_password,
        host="localhost",
        port=postgres_port
    )
    pg_cursor = pg_conn.cursor()
    ##TODO :这里会导致每次运行的时候直接在数据库后面又导入一边数据 得加一个只在第一遍跑的时候导入的判断条件
    # for csv_file_path in tqdm(csv_file_path_all):
    #     create_table_sql=generate_create_table_query(csv_file_path, csv_file_path.split('/')[-1].split('.')[0])
    #     import_data_to_postgres(csv_file_path, create_table_sql, pg_conn, pg_cursor)
    # print('postgres successfully executed.')

    neo4j_uri = "bolt://localhost:7687"
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    # import_data_to_neo4j(neo4j_driver)

    mongo_conn = pymongo.MongoClient(f"mongodb://{mongo_username}:{mongo_password}@localhost:27017/")
    mongo_db = mongo_conn["mydatabase"]

    ### LAUNCH POSTGRES QUERIES
    for query in tqdm(queries):
        result, source = query_launcher(query, redis_conn, pg_cursor)


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





