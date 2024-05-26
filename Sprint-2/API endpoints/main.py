from fastapi import FastAPI

from content_based_recommender import content_recommender
from collaborative_based_recommender import svd
from models import HybridInput, ContentInput, CollaborativeInput

app = FastAPI()


DEFAULT_NUM_MOVIE_RECOMMENDATION = 10


@app.get("/")
async def home():
    return "Recommender System"


@app.post("/content")
async def get_content_recommendation(input_data: ContentInput):
    input_data_dict = input_data.model_dump()
    title = input_data_dict["title"]
    num_movie = input_data_dict.get("num_movie", DEFAULT_NUM_MOVIE_RECOMMENDATION)
    try:
        result = content_recommender(title, num_movie)
    except KeyError:
        return {"error": "title not found in database"}
    return result


@app.post("/collaborative")
async def get_collaborative_recommendation(input_data: CollaborativeInput):
    input_data_dict = input_data.model_dump()
    user_id = input_data_dict["user_id"]
    movie_id = input_data_dict["movie_id"]
    prediction = svd.predict(user_id, movie_id)
    result = {
        "uid": prediction.uid,
        "iid": prediction.iid,
        "est": prediction.est,
        "details": prediction.details,
    }
    return result


@app.post("/hybrid")
async def get_hybrid_recommendation(input_data: HybridInput):
    # def hybrid_model(user_id, title, num_movie=30):
    input_data_dict = input_data.model_dump()
    user_id = input_data_dict["user_id"]
    title = input_data_dict["title"]
    num_movie = input_data_dict.get("num_movie", DEFAULT_NUM_MOVIE_RECOMMENDATION)
    movies_list = content_recommender(title, num_movie=num_movie)
    movies_id = list(movies_list.index)
    rank = []
    for movie_id in movies_id:
        score = svd.predict(user_id, movie_id).est
        rank.append((score, movies_list[movie_id]))

    result = sorted(rank, key=lambda x: x[0], reverse=True)
    return result
