

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).parent / "movies.csv"
TOP_N = 10

REQUIRED_COLUMNS = ["title", "genres", "overview", "keywords"]

def load_movies(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    movies = pd.read_csv(file_path)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in movies.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(missing)}"
        )

    for column in REQUIRED_COLUMNS:
        movies[column] = (
            movies[column].fillna("").astype(str)
        )

    movies = movies[
        movies["title"].str.strip().ne("")
    ].copy()

    movies = movies.drop_duplicates(
        subset=["title"], keep="first"
    ).reset_index(drop=True)

    if movies.empty:
        raise ValueError("No valid movies found.")

    return movies

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class MovieRecommender:

    def __init__(self, movies):
        self.movies = movies.copy()

        self.movies["combined_features"] = (
            self.movies["genres"] + " "
            + self.movies["overview"] + " "
            + self.movies["keywords"]
        ).apply(clean_text)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )

        self.feature_matrix = self.vectorizer.fit_transform(
            self.movies["combined_features"]
        )

        self.similarity_matrix = cosine_similarity(
            self.feature_matrix,
            dense_output=False
        ).tocsr()

        self.title_lookup = {}

        for index, title in enumerate(self.movies["title"]):
            key = title.casefold()
            self.title_lookup.setdefault(key, []).append(index)

    def search_movies(self, query, limit=10):
        query = query.strip().casefold()

        if not query:
            return []

        matches = [
            title for title in self.movies["title"]
            if query in title.casefold()
        ]

        return matches[:limit]

    def recommend(self, title, top_n=10):
        key = title.strip().casefold()

        if key not in self.title_lookup:
            matches = self.search_movies(title)

            if matches:
                print("\nMovie not found. Did you mean:")
                for match in matches:
                    print(f"- {match}")
            else:
                print("\nMovie not found in the dataset.")

            return None

        movie_index = self.title_lookup[key][0]

        scores = self.similarity_matrix.getrow(
            movie_index
        ).toarray().ravel()

        ranked_indices = scores.argsort()[::-1]
        recommendations = []

        for index in ranked_indices:
            if index == movie_index:
                continue

            recommendations.append({
                "title": self.movies.iloc[index]["title"],
                "genres": self.movies.iloc[index]["genres"],
                "similarity": float(scores[index])
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    def display_recommendations(self, title, top_n=10):
        results = self.recommend(title, top_n)

        if results is None:
            return

        if not results:
            print("No other movies available.")
            return

        print(f"\nRecommendations for '{title}':")
        print("-" * 50)

        for rank, movie in enumerate(results, start=1):
            print(f"\n{rank}. {movie['title']}")
            print(f"   Genres: {movie['genres'] or 'Unknown'}")
            print(
                f"   Similarity: "
                f"{movie['similarity'] * 100:.2f}%"
            )

def main():
    try:
        movies = load_movies(DATA_FILE)
        recommender = MovieRecommender(movies)

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"Unable to initialize the recommender: {error}")
        sys.exit(1)

    print("=" * 50)
    print("       MOVIE RECOMMENDATION SYSTEM")
    print("=" * 50)
    print(f"Loaded {len(movies)} movies successfully.")

    while True:
        print("\n1. Recommend movies")
        print("2. Search movies")
        print("3. Show all movies")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            title = input(
                "Enter a movie you like: "
            ).strip()

            if title:
                recommender.display_recommendations(
                    title, TOP_N
                )
            else:
                print("Please enter a movie title.")

        elif choice == "2":
            query = input(
                "Enter a movie title or keyword: "
            ).strip()

            matches = recommender.search_movies(
                query, limit=20
            )

            if matches:
                print("\nMatching movies:")
                for match in matches:
                    print(f"- {match}")
            else:
                print("No matching movies found.")

        elif choice == "3":
            print("\nAvailable movies:")

            for title in sorted(
                movies["title"].tolist(),
                key=str.casefold
            ):
                print(f"- {title}")

        elif choice == "4":
            print("Thank you for using the Movie Recommender!")
            break

        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
