from docker import connect_and_import_data, read_config
import pandas as pd
import random
import json

# Functions for generating rules
def generate_boolean():
    """ Generate a random boolean value. """
    return random.choice([True, False])

def generate_date():
    """ Generate a random date in YYYY-MM-DD format. """
    year = random.randint(2000, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Simplified, considering all months have 28 days
    return f"{year}-{month:02d}-{day:02d}"

def generate_rating():
    pass

def generate_positive_ratio():
    pass

def generate_price():
    """ Generate a random price between 0 and 100. """
    return round(random.uniform(0, 100), 2)

def generate_discount():
    """ Generate a random discount percentage between 0% and 100%. """
    return round(random.uniform(0, 100), 2)

def generate_user_reviews():
    """ Generate a random number of user reviews between 0 and 10,000. """
    return random.randint(0, 10000)

def generate_random_game():
    """ Generate a random game entry with similar structure to the original dataset. """
    return {
        'app_id': random.randint(100000, 999999),  # Randomly generated application ID
        'title': "Random Game " + str(random.randint(1, 1000)),  # Random title
        'date_release': generate_date(),
        'win': generate_boolean(),
        'mac': generate_boolean(),
        'linux': generate_boolean(),
        'rating': generate_rating(),
        'positive_ratio': generate_positive_ratio(),
        'user_reviews': generate_user_reviews(),
        'price_final': generate_price(),
        'price_original': generate_price(),
        'discount': generate_discount(),
        'steam_deck': generate_boolean()
    }

def insert_game(query, random_game, pg_cursor, pg_conn):
    # execute the query
    pg_cursor.execute(query, (
        random_game['app_id'],
        random_game['title'],
        random_game['win'],
        random_game['mac'],
        random_game['linux'],
        random_game['user_reviews'],
        random_game['price_final'],
        random_game['price_original'],
        random_game['discount'],
        random_game['steam_deck']
    ))

    pg_conn.commit()

def match_game(query, random_game, pg_cursor):
    pg_cursor.execute(query, (
        random_game['win'],
        random_game['mac'],
        random_game['linux'],
        random_game['user_reviews'],
        random_game['price_final'],
        random_game['price_original'],
        random_game['discount'],
        random_game['steam_deck'],
        random_game['app_id']
    ))
    similar_games = pg_cursor.fetchall()
    return similar_games

def create_relationships(driver, game_data, similarities):
    with driver.session() as session:
        session.run("""
            CREATE (g:Game {app_id: $app_id, title: $title, win: $win, mac: $mac, 
            linux: $linux, user_reviews: $user_reviews, price_final: $price_final, 
            price_original: $price_original, discount: $discount, steam_deck: $steam_deck})
            """, game_data)
        for _, game_id2, score in similarities:
            session.run("""
                       MATCH (g1:Game {app_id: $game_id1}), (g2:Game {app_id: $game_id2})
                       MERGE (g1)-[r:SIMILAR]->(g2)
                       SET r.score = $score
                       """, {"game_id1": game_data['app_id'], "game_id2": game_id2, "score": score})
def find_similar_games_neo4j(driver, game_data):
    with driver.session() as session:
        # 执行查询并创建关系
        result = session.run("""
        MATCH (original:Game {app_id: $app_id}), (g:Game)
        WHERE g.app_id <> $app_id
        WITH original, g, abs((CASE g.win WHEN true THEN 1 ELSE 0 END) - (CASE $win WHEN true THEN 1 ELSE 0 END)) +
             abs((CASE g.mac WHEN true THEN 1 ELSE 0 END) - (CASE $mac WHEN true THEN 1 ELSE 0 END)) +
             abs((CASE g.linux WHEN true THEN 1 ELSE 0 END) - (CASE $linux WHEN true THEN 1 ELSE 0 END)) +
             abs(g.user_reviews - $user_reviews) + abs(g.price_final - $price_final) AS similarity
        ORDER BY similarity
        LIMIT 10
        MERGE (original)-[r:NEO4J_SEARCH]->(g)
        SET r.similarity = similarity
        RETURN g.app_id AS similar_game_id, similarity
        """, {
            'app_id': game_data['app_id'],
            'win': game_data['win'],
            'mac': game_data['mac'],
            'linux': game_data['linux'],
            'user_reviews': game_data['user_reviews'],
            'price_final': game_data['price_final']
        })

        similar_games = []
        for record in result:
            if record["similar_game_id"] is not None:
                similar_games.append({
                    "similar_game_id": record["similar_game_id"],
                    "similarity": record["similarity"]
                })

        if not similar_games:
            print("No similar games found.")
        else:
            print("Similar games found:", similar_games)

        return similar_games


if __name__ == "__main__":
    csv_file_paths = ['./dataset/games.csv', './dataset/recommendations.csv', './dataset/users.csv']
    json_path = './dataset/games_metadata.json'
    config_path = './personal_config/sihan_config.json'
    config = read_config(config_path)
    (redis_conn, pg_conn, pg_cursor,
     neo4j_driver, mongo_collection) = connect_and_import_data(config, csv_file_paths, json_path)

    random_game = generate_random_game()
    print(random_game)

    insert_query = """
    INSERT INTO games (app_id, title, win, mac, linux, user_reviews, price_final, price_original, discount, steam_deck)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
    insert_game(insert_query, random_game, pg_cursor, pg_conn)

    similarity_query = """
    SELECT g.app_id, g.positive_ratio,
           (ABS(g.win::int - %s::int) * 5 +
            ABS(g.mac::int - %s::int) * 5 +
            ABS(g.linux::int - %s::int) * 3 + 
            ABS(g.user_reviews - %s) * 30 +
            ABS(g.price_final - %s) * 20 +
            ABS(g.price_original - %s) * 20 +
            ABS(g.discount - %s) * 15 +
            ABS(g.steam_deck::int - %s::int) * 2)/100 AS similarity_score
    FROM games g
    WHERE g.app_id <> %s
    ORDER BY similarity_score
    LIMIT 25;"""

    ## LAUNCH POSTGRES QUERIES
    res = match_game(similarity_query, random_game, pg_cursor)
    print(res)

    # calculate the weighted balance
    weighted_sum_with_none = sum(score * (1 / weight) for _, score, weight in res if score is not None)
    total_weight_with_none = sum(1 / weight for _, score, weight in res if score is not None)
    weighted_average = weighted_sum_with_none / total_weight_with_none if total_weight_with_none else None
    print('The predicted positive ratio of this new game is {0}'.format(weighted_average))

    similarities = [(random_game['app_id'], row[0], row[2]) for row in res]
    create_relationships(neo4j_driver, random_game, similarities)

    # Neo4j - Find similar games
    similar_games_neo4j = find_similar_games_neo4j(neo4j_driver, random_game)
    print("Similar games from Neo4j:", similar_games_neo4j)


