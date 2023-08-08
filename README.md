# Movie Recommender ChatBot

## Description
This is the backend code for the DialogFlow chatbot. The backend is containerized using docker and then deployed on Heroku. The backend is written in Python and uses FastAPI as the web framework. The backend uses the MovieLens dataset and Surprise library to train the model and make predictions. 

## How to run
 - To use the bot, go to the following link: https://bot.dialogflow.com/87889b32-93af-409d-afc3-0e6d8e283a5e 
 - To get a recommendation, you can say something like "I want to watch a movie" or "Recommend me a movie". 
 
 - The bot will then ask you about your ID. If you don't have an ID the bot will ask you to rate a few movies and then give you a recommendation based on your ratings.