import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of users, items, and ratings
data = {
    'UserID': [1, 1, 2, 2, 3],
    'ItemID': [101, 102, 101, 103, 102],
    'Rating': [5, 3, 4, 2, 5]
}

df = pd.DataFrame(data)

# Pivot the dataset to create the user-item matrix (users as rows, items as columns)
user_item_matrix = df.pivot_table(index='UserID', columns='ItemID', values='Rating')
user_item_matrix = user_item_matrix.fillna(0)  # Replace NaN with 0 (no interaction)

# Collaborative Filtering using SVD (Matrix Factorization)
svd = TruncatedSVD(n_components=2)  # Reduce to 2 latent features for simplicity
latent_matrix = svd.fit_transform(user_item_matrix)

# Reconstruct the original matrix from the latent features
reconstructed_matrix = np.dot(latent_matrix, svd.components_)

# Create a DataFrame for the reconstructed matrix
reconstructed_df = pd.DataFrame(reconstructed_matrix, columns=user_item_matrix.columns)
print("Reconstructed Matrix:")
print(reconstructed_df)

# Function to predict ratings for a given user and item
def predict_rating(user_id, item_id):
    if user_id not in user_item_matrix.index or item_id not in user_item_matrix.columns:
        return "User or Item not found in the dataset."
    user_index = user_id - 1  # Convert user ID to index (assuming 1-based indexing)
    item_index = user_item_matrix.columns.get_loc(item_id)
    predicted_rating = reconstructed_matrix[user_index, item_index]
    return predicted_rating

# Example: Predict rating for User 1 on Item 103
predicted_rating = predict_rating(1, 103)
print(f"Predicted rating for User 1 on Item 103: {predicted_rating}")

# Function to recommend top N items for a given user
def recommend_items(user_id, n=2):
    if user_id not in user_item_matrix.index:
        return "User not found in the dataset."
    user_index = user_id - 1  # Convert user ID to index (assuming 1-based indexing)
    predicted_ratings = reconstructed_matrix[user_index, :]
    recommended_items = np.argsort(predicted_ratings)[::-1][:n]  # Get indices of top N items
    return user_item_matrix.columns[recommended_items]

# Recommend top 2 items for User 1
recommended_items = recommend_items(1, n=2)
print(f"Top recommended items for User 1: {recommended_items}")

# Content-Based Filtering using item features (Genre, Director)
item_features = pd.DataFrame({
    'ItemID': [101, 102, 103],
    'Genre': ['Action', 'Comedy', 'Action'],
    'Director': ['John Doe', 'Jane Smith', 'Sam Brown']
})

# Use OneHotEncoder to convert categorical features to numerical values
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(item_features[['Genre', 'Director']])

# Create a DataFrame with the encoded features
encoded_feature_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
encoded_feature_df['ItemID'] = item_features['ItemID']
print("\nEncoded Item Features:")
print(encoded_feature_df)

# Compute similarity between items using cosine similarity
cosine_sim = cosine_similarity(encoded_feature_df.drop('ItemID', axis=1))

# Function to recommend similar items based on cosine similarity
def recommend_similar_items(item_id, n=2):
    if item_id not in item_features['ItemID'].values:
        return "Item not found in the dataset."
    item_index = item_features[item_features['ItemID'] == item_id].index[0]
    similarities = cosine_sim[item_index]
    similar_items = np.argsort(similarities)[::-1][1:n+1]  # Get top N similar items, excluding the item itself
    return item_features.iloc[similar_items]['ItemID'].values

# Recommend items similar to Item 101
similar_items = recommend_similar_items(101, n=2)
print(f"\nTop 2 similar items to Item 101: {similar_items}")