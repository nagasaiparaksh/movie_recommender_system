# ---------------------- Movie Recommender System ----------------------
# Imports
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Step 1: Load Dataset ----------------------
movies = pd.read_csv("/kaggle/input/tmdb-5000-movie-dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("/kaggle/input/tmdb-5000-movie-dataset/tmdb_5000_credits.csv")
movies = movies.merge(credits, left_on='id', right_on='movie_id')
print("Movies shape:", movies.shape)
print("Columns:", movies.columns.tolist())

# ---------------------- Step 2: Preprocess Metadata ----------------------
# Convert stringified lists/dicts to Python objects
for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Extract 'name' from genres, keywords, cast
for col in ['genres','keywords','cast']:
    movies[col] = movies[col].apply(lambda x: [i['name'].replace(" ", "").lower() for i in x] if isinstance(x, list) else [])

# Keep only top 3 cast members
movies['cast'] = movies['cast'].apply(lambda x: x[:3])

# Extract director from crew
def get_director(crew):
    for member in crew:
        if member.get('job') == 'Director':
            return member['name'].replace(" ", "").lower()
    return ''
movies['director'] = movies['crew'].apply(get_director)

# Combine into a single "tags" column
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['director'].apply(lambda x: [x] if x else [])
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

# ---------------------- Step 3: Content-Based Similarity ----------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])
similarity = cosine_similarity(vectors)

# ---------------------- Step 4: Recommendation Function ----------------------
def recommend(movie_name, top_n=10):
    movie_index = movies[movies['title_x'].str.lower() == movie_name.lower()].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
    recommended_movies = [movies.iloc[i[0]]['title_x'] for i in movie_list]
    return recommended_movies

# ---------------------- Step 5: Example ----------------------
movie_to_search = "The Dark Knight"
print(f"Top 10 recommendations for '{movie_to_search}':")
print(recommend(movie_to_search, top_n=10))

# ---------------------- Step 6: Collaborative Filtering Placeholder ----------------------
# Optional: If you have user ratings, you can use Surprise library:
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from surprise import accuracy
# ratings = pd.read_csv("ratings.csv")
# reader = Reader(rating_scale=(0.5,5))
# data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
# trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
# svd = SVD()
# svd.fit(trainset)
# predictions = svd.test(testset)
# print("Collaborative filtering RMSE:", accuracy.rmse(predictions))
