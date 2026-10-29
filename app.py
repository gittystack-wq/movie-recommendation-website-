from flask import Flask, render_template, request

from movie_recommendation import (
    load_movies,
    MovieRecommender,
    DATA_FILE,
    TOP_N
)

app = Flask(__name__)

# Load your existing movie dataset and ML model
movies_df = load_movies(DATA_FILE)
recommender = MovieRecommender(movies_df)

all_titles = sorted(
    movies_df["title"].tolist(),
    key=str.casefold
)


@app.route("/")
def home():
    return render_template(
        "index.html",
        all_titles=all_titles,
        recommendations=None,
        selected_movie="",
        error=None
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form.get("movie", "").strip()

    results = recommender.recommend(title, top_n=TOP_N)

    if results is None:
        return render_template(
            "index.html",
            all_titles=all_titles,
            recommendations=None,
            selected_movie=title,
            error="Movie not found. Please select a movie from the list."
        )

    return render_template(
        "index.html",
        all_titles=all_titles,
        recommendations=results,
        selected_movie=title,
        error=None
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)