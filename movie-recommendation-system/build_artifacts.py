"""Download the TMDB 5000 tables and build the pickles Cinematch imports.

The upstream repo gitignores artifacts/ and data_project/. This follows the
notebook in notebooks/programm.ipynb: merge movies with credits, build a tag
string, stem it, then save a count-vector cosine matrix.
"""

import ast
import pickle
import urllib.request
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_project"
ART = ROOT / "artifacts"
MOVIES_URL = (
    "https://raw.githubusercontent.com/pChitral/Movie-Recommender-System-FastAPI-React.js/"
    "main/backend/data/tmdb_5000_movies.csv"
)
CREDITS_URL = (
    "https://raw.githubusercontent.com/pChitral/Movie-Recommender-System-FastAPI-React.js/"
    "main/backend/data/tmdb_5000_credits.csv"
)


def _download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"already have {dest.name}")
        return
    print(f"downloading {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _names(raw: str) -> list[str]:
    try:
        return [item["name"] for item in ast.literal_eval(raw)]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []


def _cast(raw: str) -> list[str]:
    names = []
    try:
        for item in ast.literal_eval(raw):
            names.append(item["name"])
            if len(names) == 3:
                break
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []
    return names


def _directors(raw: str) -> list[str]:
    try:
        return [item["name"] for item in ast.literal_eval(raw) if item.get("job") == "Director"]
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return []


def _collapse(values: list[str]) -> list[str]:
    return [value.replace(" ", "") for value in values]


def build() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    movies_path = DATA / "movies.csv"
    credits_path = DATA / "credits.csv"
    _download(MOVIES_URL, movies_path)
    _download(CREDITS_URL, credits_path)

    df_path = ART / "new_df.pkl"
    sim_path = ART / "similarity.pkl"
    if df_path.is_file() and sim_path.is_file():
        print("recommendation artifacts already built")
        return

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    frame = movies.merge(credits, on="title")
    frame = frame[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
    frame["genres"] = frame["genres"].map(_names).map(_collapse)
    frame["keywords"] = frame["keywords"].map(_names).map(_collapse)
    frame["cast"] = frame["cast"].map(_cast).map(_collapse)
    frame["crew"] = frame["crew"].map(_directors).map(_collapse)
    frame["overview"] = frame["overview"].map(lambda text: text.split() if isinstance(text, str) else [])
    frame["tags"] = frame["overview"] + frame["genres"] + frame["keywords"] + frame["cast"] + frame["crew"]
    frame["tags"] = frame["tags"].map(lambda words: " ".join(words).lower())

    stemmer = PorterStemmer()

    def stem(text: str) -> str:
        return " ".join(stemmer.stem(word) for word in text.split())

    frame["tags"] = frame["tags"].map(stem)
    new_df = frame[["movie_id", "title", "tags"]].reset_index(drop=True)

    print(f"vectorizing {len(new_df)} movies")
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(new_df["tags"])
    similarity = cosine_similarity(vectors)

    ART.mkdir(parents=True, exist_ok=True)
    with df_path.open("wb") as handle:
        pickle.dump(new_df, handle)
    with sim_path.open("wb") as handle:
        pickle.dump(similarity, handle)
    print(f"wrote {df_path.name} and {sim_path.name}")


if __name__ == "__main__":
    build()
