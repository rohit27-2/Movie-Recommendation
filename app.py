from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the movie dataset
movies_data = pd.read_csv('./movies.csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into one
combined_features = movies_data[selected_features].apply(lambda x: ' '.join(x), axis=1)

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

# Cosine Similarity - getting similarity scores
similarity = cosine_similarity(feature_vector)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        
        # Find the close match for the movie name given by the user
        find_close_match = difflib.get_close_matches(movie_name, movies_data['title'], n=1)
        if find_close_match:
            close_match = find_close_match[0]
            index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]

            # Getting list of similar movies
            similarity_score = list(enumerate(similarity[index_of_the_movie]))

            # Sorting the movies based on their similarity score
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            # Get suggested movies
            suggested_movies = []
            for i, movie in enumerate(sorted_similar_movies[1:11], 1):
                index = movie[0]
                title_from_index = movies_data.loc[index, 'title']
                tagline_from_index = movies_data.loc[index, 'tagline']
                url = movies_data.loc[index, 'homepage']
                
                suggested_movies.append({
                    'rank': i,
                    'title': title_from_index,
                    'tagline': tagline_from_index,
                    'url': url
                })

            return render_template('results.html', movies=suggested_movies, input_movie=close_match)
        else:
            error = "Movie not found. Please try again with a different movie name."
            return render_template('index.html', error=error)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
