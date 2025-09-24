# Movie Recommender System – Hybrid Filtering

## Project Overview
This project implements a **Movie Recommender System** using **hybrid filtering** (content-based + collaborative filtering).  
The system provides personalized movie recommendations based on metadata like genres, keywords, cast, and director.  

**Technologies Used:**  
- Python  
- Pandas, NumPy  
- Scikit-learn (CountVectorizer, cosine_similarity)  
- Optional: Surprise library for collaborative filtering  

---

## Dataset
**TMDB 5000 Movie Dataset** (from Kaggle)  
- `tmdb_5000_movies.csv` – Metadata for ~5,000 movies  
- `tmdb_5000_credits.csv` – Cast and crew information  

Dataset link: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## Methodology

1. **Data Preprocessing**  
   - Merged `movies` and `credits` datasets.  
   - Converted stringified JSON columns (`genres`, `keywords`, `cast`, `crew`) to Python objects.  
   - Extracted top 3 cast members and director.  
   - Created a combined `tags` column from genres, keywords, cast, and director.  

2. **Content-Based Filtering**  
   - Vectorized `tags` using `CountVectorizer`.  
   - Computed cosine similarity between all movies.  
   - Recommender function retrieves top N similar movies.

3. **Collaborative Filtering (Optional)**  
   - Can be added using user ratings and the Surprise library (SVD or other algorithms).  

---

## Usage

```python
from recommender import recommend

movie_name = "The Dark Knight"
recommendations = recommend(movie_name, top_n=10)
print(recommendations)
