import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import csv
import redis
import psycopg2
import random
import time
from datetime import datetime 
import json
import re
import pymongo
from neo4j import GraphDatabase
from tqdm import tqdm
import docker
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from textacy import preprocessing
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.preprocessing import normalize
import subprocess
from pymongo import MongoClient
import math
import umap.umap_ as umap
import json


subprocess.run(["python", "docker.py"])

#exec(open("docker.py").read())

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def process(data):   
    data = data.drop_duplicates()
    data.dropna(axis=1)
    data = data.rename(columns = {0:'Description',1:'app_id'})
    return data 

def import_mongo_data(config):
    Mongo = MongoClient(host='localhost', port=27017, username=config['mongo']['username'], password=config['mongo']['password'])
    mongo_collection = Mongo[config['mongo']['database_name']][config['mongo']['collection_name']]
    return mongo_collection

def create_wordcloud(data, column, save_path='./game_topic_result'):
    words = data[column].values 
    wordcloud = WordCloud().generate(str(words))

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(save_path + '/wordcloud.png', bbox_inches='tight')

def stopword_data(doc,column='Description'): 
    # Create a DataFrame from the 'description' field of each document
    #descriptions = pd.DataFrame([document.get('description', '') for document in doc])
    
    descriptions = pd.DataFrame({0:[document.get('description') for document in doc.find() if document.get('description', '')], 1: [document.get('app_id') for document in doc.find() if document.get('description', '')]})

    # Process the DataFrame
    data = process(descriptions)

    # Check if the specified column exists in the DataFrame
    if column in data.columns:
        data = data[data[column].str.strip().astype(bool)]
        
        data = data[data[column].apply(lambda x: bool(re.search('[a-zA-Z0-9]', x)))]

        stop_words = [stopwords.words('english')]
        stemmer = SnowballStemmer('english')

        data[column] = data[column].str.lower()
        data[column] = data[column].apply(lambda x: ' '.join([i for i in x.split() if i not in stop_words]))
        data[column] = data[column].apply(lambda x: preprocessing.remove.punctuation(x))
        data[column] = data[column].apply(lambda x: preprocessing.normalize.whitespace(x))
        data[column] = data[column].apply(lambda x: preprocessing.replace.hashtags(x))
        data[column] = data[column].apply(lambda x: stemmer.stem(x))
    return data

def column_info(data, column, save_path):
    
    print(f"Info. about {column} columns is as follows:",'\n')
    
    data['chars'] = data[column].str.len()
    
    a = data['chars'].mean()
    
    data['words'] = data[column].str.split().str.len()
    b = data['words'].mean()
    
    c = data['words'].max()
    
    d = data['words'].min()
    
    #plt.barh(width = [a,b,c,d], y = ['Average Number of characters','Average Number of words','Maximum Number of words','Minimum Number of words'])
    #plt.title("Some insights about the Rows of the Title Column")
    #plt.xlabel("Counts")
    #plt.savefig(save_path + '/col_num.png', bbox_inches='tight')

    sns.kdeplot(data['words'], fill = True, color = 'red')
    plt.title("No. of words in Sentences")
    plt.xlabel("No. of words")
    plt.ylabel("Counts")
    plt.savefig(save_path + '/col_dist.png', bbox_inches='tight')

def fit_model(data):
    data['Description']=data['Description'].apply(lambda x: word_tokenize(x))
    model_ft = FastText(data['Description'], min_count=2, vector_size=100, window=3)
    return model_ft

def embedding_model(model,data):
    embeddings = []
    for i in data['Description']:
        embedding = np.mean([model_ft.wv[word] for word in i], axis = 0)
        embeddings.append(embedding)
    embeddings_array = np.array(embeddings)


    #umap_emb = umap.UMAP(n_components=2, random_state=42)
    #embeddings_umap = umap_emb.fit_transform(embeddings_array)

    #plt.figure(figsize=(8, 6))
    #plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1])
    #plt.title('UMAP Visualization of Embeddings')
    #plt.xlabel('UMAP Dimension 1')
    #plt.ylabel('UMAP Dimension 2')
    #plt.savefig( './game_topic_result/embedding.png', bbox_inches='tight')
    
    return embeddings

def fit_lda_model(data,input,n_components,num):
    embeddings = [emb for emb in input if emb is not None]
    embeddings = [element for element in embeddings if element.shape == (100,)]

    embeddings = np.array(embeddings)
    embeddings = normalize(embeddings)
    embeddings[embeddings < 0] = 0

    lda = LatentDirichletAllocation(n_components, learning_method = 'batch')
    topics = lda.fit_transform(embeddings)
    #for i, j in enumerate(lda.components_):
        #top_titles = data.iloc[np.argsort(j)[-5:]]['Description']
    text_data = data['Description']
    vocab = set()
    for text in text_data:
        for word in text:
            vocab.add(word)
    vocab = list(vocab)
    topic_keywords={}
    for i, top in enumerate(lda.components_):
        top_idx = top.argsort()[:-num - 1:-1]
        top_words = [vocab[j] for j in top_idx]
        topic_keywords[f'Topic_{i + 1}'] = top_words
    return topics, topic_keywords #,top_titles

def import_df_to_neo4j(df, neo4j_driver):
    game=pd.read_csv('./dataset/games.csv')
    df=pd.merge(df, game, on='app_id', how='inner') 
    with neo4j_driver.session() as session:
        for index, row in df.iterrows():
            # Extract topic related data
            description = row['Description']
            topic = row['topic']
            topic_name = row['topic_name']
            app_id = row['app_id']  # Assuming 'app_id' is the column in your DataFrame

            # Create or update the Topic node
            cypher_query_topic = (
                "MERGE (t:Topic {"
                "description: $description, "
                "topic: $topic, "
                "topic_name: $topic_name"
                "})"
            )
            session.run(cypher_query_topic, description=description, topic=topic, topic_name=topic_name)

            # Merge the Game node based on 'app_id'
            cypher_query_merge_game = (
                "MERGE (g:Game {app_id: $app_id})"
            )
            session.run(cypher_query_merge_game, app_id=app_id)

            # Create relationship and connect properties
            cypher_query_relation = (
                "MATCH (t:Topic {description: $description}), (g:Game {app_id: $app_id}) "
                "MERGE (t)-[:RELATED_TO]->(g) "
                "ON CREATE SET g.title = $title, g.rating = $rating, "
                "g.positive_ratio = $positive_ratio, g.price_final = $price_final, "
                "g.discount = $discount "
                "ON MATCH SET g.title = $title, g.rating = $rating, "
                "g.positive_ratio = $positive_ratio, g.price_final = $price_final, "
                "g.discount = $discount"
            )
            session.run(cypher_query_relation, description=description, app_id=app_id,
                       title=row['title'], rating=row['rating'], positive_ratio=row['positive_ratio'],
                       price_final=row['price_final'], discount=row['discount'])
    
    print('Data imported')

def link_games_to_topics_batch(df_batch, neo4j_driver, threshold_count):
    with neo4j_driver.session() as session:
        # Prepare batch data for the query
        batch_data = [{'app_id': row['app_id'], 'description': row['Description']} 
                      for index, row in df_batch.iterrows()]

        # Optimized Cypher query
        cypher_query = (
            "UNWIND $batchData AS row "
            "MATCH (g:Game {app_id: row.app_id}) "
            "MATCH (t:Topic {description: row.description}) "
            "WITH g, t, count(*) as rel_count "
            "WHERE rel_count > $threshold_count "
            "MERGE (g)-[:RELATED_TO]->(t);")

        # Run the Cypher query with the batch data and threshold
        session.run(cypher_query, batchData=batch_data, threshold_count=threshold_count)

    print('Game-Topic relationships created for batch')

def link_games_to_topics(df, neo4j_driver, threshold_percentage=10, batch_size=500):
    num_batches = math.ceil(df.shape[0] / batch_size)
    threshold_count = int(threshold_percentage / 100 * df.shape[0])

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, df.shape[0])
        batch_df = df.iloc[start_idx:end_idx]
        
        link_games_to_topics_batch(batch_df, neo4j_driver, threshold_count)

    print('Game-Topic relationships created')

def clear_memory_except_topic_and_game(driver):
    with driver.session() as session:
        cypher_query = (
                "MATCH (n) "
                "WHERE NOT n:Topic AND NOT n:Game "
                "DETACH DELETE n"
            )
        session.run(cypher_query)
    print('cache cleared')



if __name__ == "__main__":
    config_path = './personal_config/Jiacheng_config.json'
    config=read_config(config_path)
    neo4j_uri = "bolt://localhost:7687"
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(config['neo4j']['username'], config['neo4j']['password']))

    documents = import_mongo_data(config)

    wordcloud = pd.DataFrame({'0':[document.get('description') for document in documents.find() if document.get('description', '')]})

    #create_wordcloud(wordcloud, '0',save_path='/Users/chiuchiu/Desktop')
    #column_info(wordcloud, '0', '/Users/chiuchiu/Desktop')

    

    final_data= stopword_data(documents)

    #create_wordcloud(final_data, 'Description')
    #column_info(final_data, 'Description','./game_topic_result')

    model_ft =fit_model(final_data)
    embeddings =embedding_model(model_ft,final_data)
    
    topics, keywords =fit_lda_model(final_data, embeddings, 10, 20)

    #file_path = './game_topic_result/my_dictionary.json'

    #with open(file_path, 'w') as json_file:
        #json.dump(keywords, json_file)

    
    final_data['topic']=topics.argmax(axis=1)+1

    final_data['topic_name'] = final_data['topic'].map(dict(zip({1,2,3,4,5,6,7,8,9,10}, keywords.values())))

    final_data.to_csv('./game_topic_result/data.csv')

    import_df_to_neo4j(final_data, neo4j_driver)
    clear_memory_except_topic_and_game(neo4j_driver)
    link_games_to_topics(final_data, neo4j_driver, threshold_percentage=10, batch_size=500)
