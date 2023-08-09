from typing import Dict, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
from random import randint

# The dots before the model module are important for uvicorn to find the module
from .model import update_model_with_ratings, recommend, select_random_movie, get_movie_name
from .model import unique_user_ids

# import sys
# from fastapi.logger import logger
# from pyngrok import ngrok

app = FastAPI()

# port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 80

# public_url = ngrok.connect(
#     80, options={"domain": "enough-enormous-skink.ngrok-free.app"}
# ).public_url
# logger.info('ngrok tunnel "{}" -> "http://0.0.0.0:{}"'.format(public_url, 80))


class Intent(BaseModel):
    displayName: str


class Request(BaseModel):
    intent: Intent
    parameters: Dict[str, Any]


session_vars = {"user_id": ""}


@app.post("/")
async def predict(queryResult: Request = Body(..., embed=True)):
    intent = queryResult.intent.displayName
    print(intent)

    # The user ID will either be in the session variables or in the parameters
    # First, we check if it's in the parameters, otherwise we get from the session variables

    # user_id = queryResult.parameters.get("uid", session_vars["user_id"])
    if queryResult.parameters.get("uid"):
        session_vars["user_id"] = queryResult.parameters.get("uid", "")
        user_id = session_vars["user_id"]
    else:
        user_id = session_vars.get("user_id", "")

    if intent == "GetID":
        return handel_get_id()

    elif intent == "CheckUserID":
        return handel_check_user_id(queryResult, user_id)

    elif intent == "RateMovie":
        return handle_rate_movie(queryResult, user_id)

    elif intent == "Recommend-HasID":
        return handle_recommendation(user_id)

    else:
        return {"fulfillmentText": "I don't understand"}


# Checks for the ID in the session variables and returns it if it exists
def handel_get_id():
    if session_vars["user_id"]:
        return {
                "fulfillmentText": f"Your user ID is {int(session_vars['user_id'])}."
            }
    else:
        return {
                "fulfillmentText": "You did not share your ID with me this session."
            }


def handel_check_user_id(queryResult, user_id):
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

def handle_new_user():
    new_user_id = generate_new_user_id()
    session_vars["user_id"] = new_user_id
    movie_id = select_random_movie()
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

def handle_rate_movie(queryResult, user_id):
    if user_id:
        movie_id = select_random_movie()
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

def handle_recommendation(user_id):
    if user_id in unique_user_ids:
        recommendation = recommend(user_id)
        text = "Here are some movies you might like: \n"
        text += "\n".join(recommendation)
        return {"fulfillmentText": text}
    else:
        return {"fulfillmentText": "User ID not found."}

def generate_new_user_id():
    new_user_id = randint(10000, 100000)
    while new_user_id in unique_user_ids:
        new_user_id = randint(10000, 100000)
    unique_user_ids.append(new_user_id)
    return str(new_user_id)
