import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from docker import connect_and_import_data, read_config


def game_cluster(pg_conn):
    query = """
            SELECT app_id, positive_ratio, user_reviews, price_final, discount, win, mac, linux, steam_deck
            FROM games;
            """
    games_data = pd.read_sql(query, pg_conn)
    games_data[['win', 'mac', 'linux', 'steam_deck']] = games_data[['win', 'mac', 'linux', 'steam_deck']].astype(int)
    games_data['user_reviews'] = games_data['user_reviews'].astype(int)
    games_data['discount'] = games_data['discount'].astype(float)
    games_data['positive_ratio'] = games_data['positive_ratio'].astype(float)
    games_data['price_final'] = games_data['price_final'].astype(float)
    games_data['user_reviews'] = games_data['user_reviews'].astype(int) 

    features_for_pca = games_data.drop(['app_id'], axis=1)

    # PCA pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 5))  # n_components has been tested
    ])

    # Fit and transform the data
    principal_components = pipeline.fit_transform(features_for_pca)

    print("PCA Components:")
    print(pipeline.named_steps['pca'].components_)
    print("Explained Variance Ratio:")
    print(pipeline.named_steps['pca'].explained_variance_ratio_)

    kmeans = KMeans(n_clusters=10, random_state=42)
    games_data['cluster'] = kmeans.fit_predict(principal_components)

    # print("Cluster Centers:")
    # print(kmeans.cluster_centers_)

    print(games_data.head())

    games_data.to_csv('./game_cluster_result/clustered_games.csv', index=False)
    games_data.to_csv('./neo4j_import/clustered_games.csv', index=False)

def game_cluster_import_to_neo4j(driver):
    with driver.session() as session:
        session.run('''match(n)  detach  delete  n''')

        session.run('''LOAD CSV WITH HEADERS FROM 'file:///clustered_games.csv' AS row
                       MERGE (cluster:Cluster {id: toInteger(row.cluster)});''')
        print('cluster node created')

        session.run('''LOAD CSV WITH HEADERS FROM 'file:///clustered_games.csv' AS row
                      MERGE (game:Game {id: row.app_id})
                      ON CREATE SET 
                                    game.positive_ratio = toFloat(row.positive_ratio),
                                    game.user_reviews = toInteger(row.user_reviews),
                                    game.price_final = toFloat(row.price_final),
                                    game.discount = toFloat(row.discount),
                                    game.win = toInteger(row.win),
                                    game.mac = toInteger(row.mac),
                                    game.linux = toInteger(row.linux),
                                    game.steam_deck = toInteger(row.steam_deck)
                      MERGE (cluster:Cluster {id: toInteger(row.cluster)})
                      CREATE (game)-[:BELONGS_TO]->(cluster);''')
        print('game node created and relationship created')


if __name__ == "__main__":
    csv_file_paths = ['./dataset/games.csv', './dataset/recommendations.csv', './dataset/users.csv']
    json_path = './dataset/games_metadata.json'
    config_path = './personal_config/xueqi_config.json'
    config = read_config(config_path)
    (redis_conn, pg_conn, pg_cursor,
     neo4j_driver, mongo_collection) = connect_and_import_data(config, csv_file_paths, json_path)

    game_cluster(pg_conn)
    game_cluster_import_to_neo4j(neo4j_driver)

    # neo4j query
    # MATCH (g:Game)-[:BELONGS_TO]->(c:Cluster) RETURN g, c