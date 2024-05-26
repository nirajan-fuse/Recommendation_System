import pickle


with open("./data/cosine_similarities.pkl", "rb") as f:
    cosine_similarities = pickle.load(f)

with open("./data/indices.pkl", "rb") as f:
    indices = pickle.load(f)

with open("./data/movie_title.pkl", "rb") as f:
    movie_title = pickle.load(f)


def content_recommender(title, num_movies=25):
    if title not in indices:
        raise KeyError("Title Not Found in database.")
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_movies + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movie_title.iloc[movie_indices]
