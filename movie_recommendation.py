import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('/content/ml-latest-small/movies.csv')

# Fill missing values
movies['genres'] = movies['genres'].fillna('')

# Convert genres into TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index
movies = movies.reset_index()

# Create title-to-index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to recommend movies
def recommend_movies(title, cosine_sim=cosine_sim):
    
    # Search similar movie names
    matches = movies[movies['title'].str.contains(title, case=False)]

    # If no movie found
    if matches.empty:
        print("Movie not found!")
        return

    # Show matching movies
    print("\nMatching Movies:")
    for i, movie in enumerate(matches['title'].head(10)):
        print(f"{i+1}. {movie}")

    # User selects movie
    selected_movie = input("\nCopy and enter exact movie name: ")

    # Check again
    if selected_movie not in indices:
        print("Exact movie name not found!")
        return

    # Get movie index
    idx = indices[selected_movie]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Top 10 similar movies
    sim_scores = sim_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Print recommendations
    print("\nRecommended Movies:\n")

    for movie in movies['title'].iloc[movie_indices]:
        print(movie)

# Take user input
movie_name = input("Enter movie name: ")

# Call function
recommend_movies(movie_name)
