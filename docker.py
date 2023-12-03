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
    pg_cursor.execute("SET search_path TO public")
    pg_conn.commit()
    # for this step, set the path to your own destination.

    pg_cursor.execute(create_table_sql)
    pg_conn.commit()

    # Get the table name from the CSV file name
    table_name = csv_file_path.split('/')[-1].split('.')[0]

    # Check if the table exists and has data
    pg_cursor.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='{table_name}');")
    table_exists = pg_cursor.fetchone()[0]

    if table_exists:
        pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = pg_cursor.fetchone()[0]
        if count > 0:
            # Table exists and has data, skip data import
            print(f"Table '{table_name}' already exists and has data. Skipping data import.")
            return

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
        session.run('''match(n)  detach  delete  n''')

        # 导入用户数据
        session.run(''' LOAD CSV WITH HEADERS FROM 'file:///games.csv' AS row
                        CREATE (:Game {
                                        app_id: toInteger(row.app_id),
                                        title: row.title,
                                        date_release: row.date_release,
                                        win: row.win = 'True',
                                        mac: row.mac = 'True',
                                        linux: row.linux = 'True',
                                        rating: row.rating,
                                        positive_ratio: toInteger(row.positive_ratio),
                                        user_reviews: toInteger(row.user_reviews),
                                        price_final: toFloat(row.price_final),
                                        price_original: toFloat(row.price_original),
                                        discount: toFloat(row.discount),
                                        steam_deck: row.steam_deck = 'True'});''')
        print('users file imported')

        # 导入游戏数据
        session.run(''' LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
                        CREATE (:User {
                                       user_id: toInteger(row.user_id),
                                       products: toInteger(row.products),
                                       reviews: toInteger(row.reviews)});''')
        print('games file imported')

        session.run('''LOAD CSV WITH HEADERS FROM 'file:///recommendations.csv' AS row
                        CREATE (:Recommendation {
                                       helpful: toInteger(row.helpful),
                                       funny: toInteger(row.funny),
                                       date: row.date,
                                       is_recommended: row.is_recommended = 'True',
                                       hours: toFloat(row.hours),
                                       review_id: toInteger(row.review_id),
                                       user_id: toInteger(row.user_id),
                                       game_id: toInteger(row.app_id)});''')
        print('recommendations file imported')

        session.run('''MATCH (user:User), (rec:Recommendation {user_id: user.user_id})
                       CREATE (user)-[:HAS_RECOMMENDED]->(rec);''')
        print('user to recommendation relationships created')

        session.run(''' MATCH (game:Game), (rec:Recommendation {game_id: game.app_id})
                        CREATE (rec)-[:RECOMMENDS]->(game);''')
        print('recommendation to game relationships created')

        session.run('''MATCH (u:User), (r:Recommendation), (g:Game)
                       WHERE r.user_id = u.user_id AND r.game_id = g.app_id
                       CREATE (u)-[:HAS_RECOMMENDED]->(r)-[:RECOMMENDS]->(g);''')

        print('All data imported and relationships created')


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

def import_json_to_mongodb(json_file_path, db_name, collection_name, host='localhost', port=27017, username=None, password=None):    
    # Establish a connection to the MongoDB server
    client_kwargs = {'host': host, 'port': port}
    if username and password:
        client_kwargs['username'] = username
        client_kwargs['password'] = password
        client_kwargs['authSource'] = 'admin'  # Default authSource is 'admin'

    client = pymongo.MongoClient(**client_kwargs)


    # Select the database and collection, IF NOT EXISTS, MONGODB AUTOMATICALLY CREATE ONE.
    db = client[db_name]    
    collection = db[collection_name]
    
        # Check if the collection already has data
    if collection.estimated_document_count() > 0:
        # Collection exists and has data, skip data import
        print(f"Collection '{db_name}.{collection_name}' already exists and has data. Skipping data import.")
        return collection
    
    # Load the JSON file
    with open(json_file_path, 'r',encoding='utf-8') as file:
        data_string = file.read()
        # Assuming that each JSON object is separated by a newline
        json_objects = data_string.split('\n')
        for json_object in tqdm(json_objects):
            if json_object.strip():  # Skip empty lines
                data = json.loads(json_object)
                collection.insert_one(data)
    
    print(f"Data from {json_file_path} has been imported to the '{db_name}.{collection_name}' collection.")
    return collection



# def recommend_games():
    # use Neo4j to generate game recommendations.
    #return jsonify(recommendation_result)


########## game for single search ##########
def game_search(config, query_params, pg_conn):
    mongo_query_params = {}
    pg_query_params = {}

    for key, value in query_params.items():
        if key in ['tags_include_any', 'tags_include_all', 'description']:
            mongo_query_params[key] = value
        else:
            pg_query_params[key] = value

    mongo_results = execute_mongo_query(mongo_query_params, config['mongo']['database_name'], config['mongo']['collection_name'],
                                        config['mongo']['username'], config['mongo']['password']) if mongo_query_params else []
    pg_results = execute_postgres_query(pg_query_params, pg_conn) if pg_query_params else []
    return merge_results(mongo_results, pg_results)


def merge_results(mongo_results, pg_results):
    mongo_results_dict = {item['app_id']: item for item in mongo_results}
    pg_results_dict = {item['app_id']: item for item in pg_results}

    merged_results = {}

    # first, add mongo_results into res
    for app_id, mongo_result in mongo_results_dict.items():
        merged_results[app_id] = mongo_result

    # then, add update or add new pg_results into res
    for app_id, pg_result in pg_results_dict.items():
        if app_id in merged_results:
            # merge mongo & postgres
            merged_results[app_id].update(pg_result)
        else:
            # direct add postgre
            merged_results[app_id] = pg_result

    merged_results_list = list(merged_results.values())[:5] # show top10 to test
    merged_results_json = json.dumps(merged_results_list, default=str)
    return merged_results_json


def execute_mongo_query(query_params, db_name, collection_name, mongo_username, mongo_password):
    client = pymongo.MongoClient(host='localhost', port=27017, username=mongo_username, password=mongo_password, authSource='admin')
    db = client[db_name]
    collection = db[collection_name]

    query_dict = {}

    for query_type, query_param in query_params.items():
        if  query_type == 'tags_include_any':
            query_dict["tags"] = {"$in": query_param}
        elif query_type == 'tags_include_all':
            query_dict["tags"] = {"$all": query_param}
        elif query_type == 'description':
            collection.create_index([("description", pymongo.TEXT)])
            query_dict["$text"] = {"$search": query_param}

    result = list(collection.find(query_dict)) # select *
    return result


def execute_postgres_query(query_params, pg_conn):
    query_conditions = []
    for query_type, query_param in query_params.items():
        if query_type == 'app_id':
            query_conditions.append(f"app_id = {query_param}")
        elif query_type == 'title':
            query_conditions.append(f"title ILIKE '%{query_param}%'")
        elif query_type == 'date_release':
            query_conditions.append(f"date_release > '{query_param}'")
        elif query_type == 'win':
            query_conditions.append(f"win = '{query_param}'")
        elif query_type == 'mac':
            query_conditions.append(f"mac = '{query_param}'")
        elif query_type == 'linux':
            query_conditions.append(f"linux = '{query_param}'")
        elif query_type == 'positive_ratio':
            query_conditions.append(f"positive_ratio > {query_param}")
        elif query_type == 'user_reviews':
            query_conditions.append(f"user_reviews > {query_param}")
        elif query_type == 'rating':
            query_conditions.append(f"rating = '{query_param}'")
        elif query_type == 'price_final':
            query_conditions.append(f"price_final BETWEEN {query_param - 10} AND {query_param + 10}")
        elif query_type == 'price_original':
            query_conditions.append(f"price_original BETWEEN {query_param - 10} AND {query_param + 10}")
        elif query_type == 'discount':
            query_conditions.append(f"discount > {query_param}")
        elif query_type == 'steam_deck':
            query_conditions.append(f"steam_deck = {query_param}")  # Use TRUE or FALSE for the value of query_param

    if query_conditions:
        query = f"SELECT * FROM games WHERE {' AND '.join(query_conditions)};"
        with pg_conn.cursor() as cursor:
            cursor.execute(query)
            records = cursor.fetchall()
            return [dict(zip([col[0] for col in cursor.description], row)) for row in records]  # 返回列表而不是 JSON 字符串
    else:
        return []


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def connect_and_import_data(config, csv_file_paths, json_path):
    ### IMPORT DATA TO DIFFERENT DATABASES
    redis_conn = redis.StrictRedis(host='localhost', port=config['redis']['port'], db=config['redis']['database_name'])

    pg_conn = psycopg2.connect(
        dbname=config['postgres']['database_name'],
        user=config['postgres']['username'],
        password=config['postgres']['password'],
        host="localhost",
        port=config['postgres']['port']
    )
    pg_cursor = pg_conn.cursor()

    for fp in tqdm(csv_file_paths):
         create_table_sql=generate_create_table_query(fp, fp.split('/')[-1].split('.')[0])
         import_data_to_postgres(fp, create_table_sql, pg_conn, pg_cursor)
    print('postgres successfully executed.')

    neo4j_uri = "bolt://localhost:7687"
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(config['neo4j']['username'], config['neo4j']['password']))
    import_data_to_neo4j(neo4j_driver)

    ##print(json_path, config['mongo']['database_name'], config['mongo']['collection_name'],config['mongo']['username'], config['mongo']['password'])

    mongo_collection = import_json_to_mongodb(json_path, db_name=config['mongo']['database_name'], collection_name=config['mongo']['collection_name'],
                                              username= config['mongo']['username'], password= config['mongo']['password'])
    
    
    return redis_conn, pg_conn, pg_cursor, neo4j_driver, mongo_collection

# Main script execution
if __name__ == "__main__":
    csv_file_paths = ['./dataset/games.csv', './dataset/recommendations.csv', './dataset/users.csv']
    json_path = './dataset/games_metadata.json'
    config_path = './personal_config/Xueqi_config.json'
    config = read_config(config_path)

    (redis_conn, pg_conn, pg_cursor,
     neo4j_driver, mongo_collection) = connect_and_import_data(config, csv_file_paths, json_path)

    ### LAUNCH POSTGRES QUERIES
    # for query in tqdm(queries):
    #     result, source = query_launcher(query, redis_conn, pg_cursor)

    # Test cases for the single game_search function
    test_queries = [
        {'title': 'The Night Fisherman'},
        {'app_id': 1227449},
        {'price_original': 20, 'win': 'TRUE', 'tags_include_all': ['Multiplayer', 'RPG']},
        {'windows': 'TRUE', 'mac': 'TRUE', 'linux': 'TRUE'},
        {'date_released': '2020-01-01', 'tags_include_any': ['RPG', 'Strategy']},
        {'user_reviews': 50, 'tags_include_any': ['Casual', 'RPG']},
        {'positive_ratio': 80, 'user_reviews': 200, 'discount': 20}
    ]

    all_results = []

    ##### test game_search func ######
    for i, test_query in enumerate(test_queries):
        print(f"Executing test case {i+1}: {test_query}")
        result = game_search(config, test_query, pg_conn)
        all_results.append({"test_case": i+1, "query": test_query, "result": json.loads(result)})

    with open('./search_game_test_result/game_search_res_limit_10.json', 'w', encoding='utf-8') as file:
        json.dump(all_results, file, indent=4)
