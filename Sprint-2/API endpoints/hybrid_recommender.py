import pickle
from content_based_recommender import content_recommender
from collaborative_based_recommender import svd

with open("./data/movie_title.pkl", "rb") as f:
    movie_title = pickle.load(f)


def hybrid_model(user_id, title, num_movies=30):
    movies_list = content_recommender(title, num_movies=num_movies)
    movies_id = list(movies_list.index)
    rank = []
    for movie_id in movies_id:
        score = svd.predict(user_id, movie_id).est
        rank.append((score, movies_list[movie_id]))

    result = sorted(rank, key=lambda x: x[0], reverse=True)
    return result
