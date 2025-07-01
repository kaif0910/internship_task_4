from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Sample movie dataset
movies = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Tenet'],
    'description': [
        'Dreams and subconscious worlds',
        'Space travel and time dilation',
        'Batman and Joker fight',
        'Inverted time and spies'
    ]
})

# Precompute TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if title not in movies['title'].values:
        return jsonify({'error': 'Movie not found'}), 404

    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:3]
    recommended = [movies.iloc[i[0]]['title'] for i in sim_scores]

    return jsonify({'recommendations': recommended})

if __name__ == '__main__':
    app.run(debug=True)
