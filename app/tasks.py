import json
import logging
import pandas as pd
import math
import numpy as np

from jakastor.celery import app
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel

logging.basicConfig(filename="process.log", level=logging.INFO)

mapping = { 'barat': {'long': 106.826927,'lat': -6.150760}, 'utara': {'long': 106.774124, 'lat': -6.121435}, 'timur': {'long': 106.882744, 'lat': -6.230702}, 'selatan': {'long': 106.814095, 'lat': -6.300641}, 'pusat': {'long': 106.829361, 'lat': -6.173110} }

def get_dist(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2)) + (math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon/2) ** 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

@app.task
def train():
    data = pd.read_excel('dataset.xlsx')[['Restaurant ID', 'Cuisines']]
    categories = data['Cuisines']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    vectors = vectorizer.fit_transform(categories)
    cosine_sim = linear_kernel(vectors, vectors)
    np.save('cosine_sim', cosine_sim)

@app.task
def predict(min, max, loc, categories):
    cosine_sim = np.load('cosine_sim.npy')
    data = pd.read_excel('dataset.xlsx')
    # data = data[(data['Average Cost for two'] > min) & (data['Average Cost for two'] < max)]

    indices = pd.Series(data.index, index=data['Restaurant ID']).drop_duplicates()

    avg = (min + max) // 2
    data['cost_dist'] = np.abs(data['Average Cost for two'] // 2 - avg)

    pos1 = mapping[loc.lower()]
    pos2 = [mapping[i.split()[1].lower()] for i in data['City'].values]
    data['dist'] = [get_dist(pos1['lat'], pos1['long'], i['lat'], i['long']) for i in pos2]

    data2 = data[data['Cuisines'].str.contains(categories.capitalize())]

    data2 = data2.sort_values(by=['dist', 'cost_dist'], ascending=[1,1])
    
    idx = indices[data.values[0][0]]

    sim_scores = list((cosine_sim[idx]))

    data['sim'] = sim_scores

    data.sort_values(['sim', 'dist', 'cost_dist'], ascending=[0,1,1])

    return data

