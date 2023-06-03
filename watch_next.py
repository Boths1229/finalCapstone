#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def watch_next(description):
    # Read the movie descriptions from the file
    with open('movies.txt', 'r') as file:
        movie_descriptions = file.read().splitlines()

    # Add the input description to the list of movie descriptions
    movie_descriptions.append(description)

    # Convert the movie descriptions into vectors
    vectorizer = CountVectorizer().fit_transform(movie_descriptions)
    vectors = vectorizer.toarray()

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(vectors)

    # Get the index of the most similar movie to the input description
    most_similar_index = np.argmax(similarity_matrix[-1, :-1])

    # Read the movie titles from the file
    with open('movies.txt', 'r') as file:
        movie_titles = file.read().splitlines()

    # Return the title of the most similar movie
    return movie_titles[most_similar_index]

# Example usage
description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator."
next_movie = watch_next(description)
print("Next movie to watch:", next_movie)


# In[ ]:




