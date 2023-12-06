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

subprocess.run(["python", "docker.py"])

#exec(open("docker.py").read())

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def process(data):   
    data = data.drop_duplicates()
    data.dropna(axis=1)
    data = data.rename(columns = {0:'Description'})
    return data 

def import_mongo_data(config, json_path):
    mongo_collection = docker.import_json_to_mongodb(json_path, db_name=config['mongo']['database_name'], collection_name=config['mongo']['collection_name'],
                                              username= config['mongo']['username'], password= config['mongo']['password'])
    return mongo_collection

def create_wordcloud(data, column):
    words = data[column].values 
    wordcloud = WordCloud().generate(str(words))

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    plt.figure(figsize = (40,40), facecolor = None)

def stopword_data(doc, column='Description'): 
    # Create a DataFrame from the 'description' field of each document
    #descriptions = pd.DataFrame([document.get('description', '') for document in doc])
    descriptions = pd.DataFrame([document.get('description') for document in doc if document.get('description', '')])

    # Process the DataFrame
    data = process(descriptions)

    # Check if the specified column exists in the DataFrame
    if column in data.columns:
        # Check if the column has any non-empty text
        data = data[data[column].str.strip().astype(bool)]
        data = data[data[column].apply(lambda x: bool(re.search('[a-zA-Z0-9]', x)))]

        # Apply further processing to the text in the specified column
        stop_words = [stopwords.words('english')]
        stemmer = SnowballStemmer('english')

        # Additional processing steps...
        data[column] = data[column].str.lower()
        data[column] = data[column].apply(lambda x: ' '.join([i for i in x.split() if i not in stop_words]))
        data[column] = data[column].apply(lambda x: preprocessing.remove.punctuation(x))
        data[column] = data[column].apply(lambda x: preprocessing.normalize.whitespace(x))
        data[column] = data[column].apply(lambda x: preprocessing.replace.hashtags(x))
        data[column] = data[column].apply(lambda x: stemmer.stem(x))
    return data

def fit_model(data):
    data['Description']=data['Description'].apply(lambda x: word_tokenize(x))
    model_ft = FastText(data['Description'], min_count=2, vector_size=100, window=3)
    return model_ft

def embedding_model(model,data):
    embeddings = []
    for i in data['Description']:
        embedding = np.mean([model_ft.wv[word] for word in i], axis = 0)
        embeddings.append(embedding)
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


if __name__ == "__main__":
    config_path = './personal_config/Jiacheng_config.json'
    json_path = './dataset/games_metadata.json'


    documents = import_mongo_data(read_config(config_path),json_path).find()
    final_data= stopword_data(documents)

    create_wordcloud(final_data, 'Description')
    model_ft =fit_model(final_data)
    embeddings =embedding_model(model_ft,final_data)
    
    topics, keywords =fit_lda_model(final_data, embeddings, 5, 20)
    
    final_data['topic']=topics.argmax(axis=1)+1
    
    print(topics,keywords,final_data)

    







