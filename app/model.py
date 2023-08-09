from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

import pandas as pd

import pickle
import os

class Model():
    def __init__(self, model_file_path, data):
        self.model_file_path = model_file_path
        self.data = data    
        
    # Reads the model from the file
    def get_model(self):
        if os.path.exists(self.model_file_path):
            with open(self.model_file_path, "rb") as file:
                self.algo = pickle.load(file)
                return self.algo
        else:
            return self.train_model()

    # Trains the model and saves it to a file 
    def train_model(self):
        trainset, _ = self.data.get_train_test()
        self.algo = KNNBasic()
        self.algo.fit(trainset)
        with open(self.model_file_path, "wb") as file:
            pickle.dump(self.algo, file)
        return self.algo


class Data():
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.data = self.get_data()
        
    def get_data(self):
        self.data = pd.read_csv(self.data_file_path + "/data.csv")
        return self.data
    
    def get_users_ids(self):
        unique_user_ids = self.data["user_id"].unique().tolist()
        return unique_user_ids
    
    def get_train_test(self):
        reader = Reader(rating_scale=(1, 5))
        surprise_data = Dataset.load_from_df(self.data[["user_id", "movie_id", "rating"]], reader)
        train_set, test_set = train_test_split(surprise_data, test_size=0.2)
        return train_set, test_set
    def add_row(self, row):
        self.data = pd.concat([self.data, row], ignore_index=True)

    def save(self):
        self.data.to_csv(self.data_file_path + "/data.csv", index=False)
        self.data = self.get_data()
    
    def select_random_movie(self):
        return self.data.sample()["movie_id"].iloc[0]

    def get_movie_name(self, movie_id):
        return movie_id_map[movie_id_map["movie_id"] == movie_id]["movie_title"].iloc[0]
    
    def get_unique_movie_ids(self):
        return self.data["movie_id"].unique().tolist()

model_file_path = "app/trained_model.pkl"
data_file_path = "app/data"

movie_id_map = pd.read_csv(data_file_path + "/movie_id_map.csv")

data = Data(data_file_path)
algo = Model(model_file_path, data).get_model()

unique_user_ids = data.get_users_ids()


def update_model_with_ratings(user_id, user_ratings):
    for movie_id, rating in user_ratings.items():
        if movie_id is not None:
            new_row = {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": float(rating),
            }
            new_row = pd.DataFrame(new_row, index=[0])
            data.add_row(new_row)
    
    data.save()
    trainset, _ = data.get_train_test()
    algo.fit(trainset)
    with open(model_file_path, "wb") as file:
        pickle.dump(algo, file)


def recommend(user_id: int, n: int = 10):
    predictions = []
    for movie_id in data.get_unique_movie_ids():
        predictions.append(
            (
                movie_id,
                algo.predict(user_id, movie_id).est,
            )
        )
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:n]
    top_movies = [
        get_movie_name(movie_id)
        for movie_id, _ in top_movies
    ]
    return top_movies

def get_movie_name(movie_id):
    return data.get_movie_name(movie_id)

def select_random_movie():
    return data.select_random_movie()