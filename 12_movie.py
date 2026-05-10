import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url = "https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv"
df = pd.read_csv(url)

features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df["combined_features"] = df.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

similarity_scores = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(similarity_scores[movie_index]))

sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

print(f"Top 5 movies similar to {movie_user_likes}:")
for i in range(1, 6):
    print(get_title_from_index(sorted_similar_movies[i][0]))