from docker import connect_and_import_data, read_config
import pandas as pd
import random

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



if __name__ == "__main__":
    csv_file_paths = ['./dataset/games.csv', './dataset/recommendations.csv', './dataset/users.csv']
    json_path = './dataset/games_metadata.json'
    config_path = './personal_config/sihan_config.json'
    config = read_config(config_path)

    # (redis_conn, pg_conn, pg_cursor,
    #  neo4j_driver, mongo_collection) = connect_and_import_data(config, csv_file_paths, json_path)

    random_game = generate_random_game()
    print(random_game)


