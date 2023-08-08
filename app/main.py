import os
import sys

import pandas as pd
import pickle
from typing import Dict, Any
from fastapi import FastAPI, Body
from fastapi.logger import logger
from pydantic import BaseModel

# from pyngrok import ngrok

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from random import randint

app = FastAPI()

# port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 80

# public_url = ngrok.connect(
#     80, options={"domain": "enough-enormous-skink.ngrok-free.app"}
# ).public_url
# logger.info('ngrok tunnel "{}" -> "http://0.0.0.0:{}"'.format(public_url, 80))

model_file_path = "app/trained_model.pkl"
data_file_path = "app/data"

movie_id_map = pd.read_csv(data_file_path + "/movie_id_map.csv")
data = pd.read_csv(data_file_path + "/data.csv")
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[["user_id", "movie_id", "rating"]], reader)
unique_user_ids = data["user_id"].unique().tolist()

if os.path.exists(model_file_path):
    algo = pickle.load(open(model_file_path, "rb"))
else:
    trainset, _ = train_test_split(surprise_data, test_size=0.2)
    algo = KNNBasic()
    algo.fit(trainset)
    pickle.dump(algo, open(model_file_path, "wb"))


class Intent(BaseModel):
    displayName: str


class Request(BaseModel):
    intent: Intent
    parameters: Dict[str, Any]


session_vars = {}


@app.post("/")
async def predict(queryResult: Request = Body(..., embed=True)):
    # print(queryResult)
    intent = queryResult.intent.displayName
    if queryResult.parameters.get("uid"):
        session_vars["user_id"] = queryResult.parameters.get("uid", None)
        user_id = session_vars["user_id"]
    else:
        user_id = session_vars.get("user_id", None)

    if intent == "GetID":
        if session_vars["user_id"]:
            return {
                "fulfillmentText": f"Your user ID is {int(session_vars['user_id'])}."
            }
        else:
            return {
                "fulfillmentText": "You did not share your ID with me this session."
            }

    if intent == "CheckUserID":
        yesORno = queryResult.parameters.get("yesORno")
        if yesORno == "yes":
            if user_id:
                return handle_existing_user(user_id)
            else:
                return {"fulfillmentText": "Please provide a valid user ID."}
        elif yesORno == "no":
            return handle_new_user()
        else:
            return {"fulfillmentText": "I'm sorry, I didn't understand your response."}

    if intent == "RateMovie":
        if user_id:
            movie_id = data.sample()["movie_id"].iloc[0]
            if list(user_ratings.values())[0] == None:
                rating = queryResult.parameters.get("rating")
                user_ratings[list(user_ratings.keys())[0]] = rating
                return {
                    "fulfillmentText": f"What do you think of {get_movie_name(movie_id)}?"
                }
            elif len(user_ratings) < 5:
                rating = queryResult.parameters.get("rating")
                user_ratings[movie_id] = rating
                return {
                    "fulfillmentText": f"Next, what do you think of {get_movie_name(movie_id)}?"
                }
            update_model_with_ratings(user_id, user_ratings)
            return {
                "fulfillmentText": "Thank you for your ratings. Your preferences have been updated.\n\nNow whenever you give me you're ID I'll be able to recommend you movies."
            }
        else:
            return {"fulfillmentText": "Please provide a valid user ID."}

    if intent == "Recommend-HasID":
        return handle_recommendation(user_id)

    else:
        return {"fulfillmentText": "I don't understand"}


def handle_new_user():
    new_user_id = generate_new_user_id()
    session_vars["user_id"] = new_user_id
    movie_id = data.sample()["movie_id"].iloc[0]
    global user_ratings
    user_ratings = {movie_id: None}
    return {
        "fulfillmentText": f"Welcome! Your new user ID is {new_user_id}. Please rate the following movies on a scale of 1 to 5.\n\nFirstly, what do you think of {get_movie_name(movie_id)}?"
    }


def handle_existing_user(user_id):
    if user_id in unique_user_ids:
        recommendation = recommend(user_id)
        text = "Here are some movies you might like: \n"
        text += "\n".join(recommendation)
        return {"fulfillmentText": text}
    else:
        return {"fulfillmentText": "User ID not found."}


def handle_recommendation(user_id):
    if user_id in unique_user_ids:
        recommendation = recommend(user_id)
        text = "Here are some movies you might like: \n"
        text += "\n".join(recommendation)
        return {"fulfillmentText": text}
    else:
        return {"fulfillmentText": "User ID not found."}


def recommend(user_id: int, n: int = 10):
    predictions = []
    for movie_id in data["movie_id"].unique():
        predictions.append(
            (
                movie_id,
                algo.predict(user_id, movie_id).est,
            )
        )
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:n]
    top_movies = [
        movie_id_map[movie_id_map["movie_id"] == movie_id]["movie_title"].iloc[0]
        for movie_id, _ in top_movies
    ]
    return top_movies


def get_movie_name(movie_id):
    return movie_id_map[movie_id_map["movie_id"] == movie_id]["movie_title"].iloc[0]


def generate_new_user_id():
    new_user_id = randint(10000, 100000)
    while new_user_id in unique_user_ids:
        new_user_id = randint(10000, 100000)
    unique_user_ids.append(new_user_id)
    return new_user_id


def update_model_with_ratings(user_id, user_ratings):
    global data
    global algo
    for movie_id, rating in user_ratings.items():
        if movie_id is not None:
            new_row = {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": float(rating),
            }
            new_row = pd.DataFrame(new_row, index=[0])
            data = pd.concat([data, new_row], ignore_index=True)
    data.to_csv(data_file_path + "/data.csv", index=False)
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data, reader)
    trainset, _ = train_test_split(surprise_data, test_size=0.2)
    algo.fit(trainset)
    pickle.dump(algo, open(model_file_path, "wb"))
