import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

class SmartRecEngine:
    def __init__(self):
        self.users = {}
        self.items = {}
        self.user_item_matrix = None
        self.item_features = None
        
    def add_user(self, user_id, preferences):
        """Add a user with their preferences"""
        self.users[user_id] = preferences
        return user_id
    
    def add_item(self, item_id, features):
        """Add an item with its features"""
        self.items[item_id] = features
        return item_id
    
    def build_matrices(self):
        """Build user-item interaction and item feature matrices"""
        # Create item feature matrix
        item_ids = list(self.items.keys())
        if not item_ids:
            return
        
        # Assuming features are dictionaries with same keys
        feature_keys = list(self.items[item_ids[0]].keys())
        self.item_features = np.array([
            [self.items[item_id].get(key, 0) for key in feature_keys]
            for item_id in item_ids
        ])
        
    def recommend_content_based(self, item_id, top_n=5):
        """Recommend items similar to given item (content-based filtering)"""
        if item_id not in self.items:
            return []
        
        self.build_matrices()
        if self.item_features is None or len(self.item_features) == 0:
            return []
        
        item_ids = list(self.items.keys())
        item_idx = item_ids.index(item_id)
        
        # Calculate similarity between items
        similarities = cosine_similarity(
            self.item_features[item_idx].reshape(1, -1),
            self.item_features
        )[0]
        
        # Get top N similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        recommendations = [
            {
                'item_id': item_ids[idx],
                'score': float(similarities[idx]),
                'features': self.items[item_ids[idx]]
            }
            for idx in similar_indices
        ]
        
        return recommendations
    
    def recommend_for_user(self, user_id, top_n=5):
        """Recommend items for a user based on their preferences"""
        if user_id not in self.users:
            return []
        
        user_prefs = self.users[user_id]
        self.build_matrices()
        
        if self.item_features is None or len(self.item_features) == 0:
            return []
        
        # Convert user preferences to vector
        item_ids = list(self.items.keys())
        feature_keys = list(self.items[item_ids[0]].keys())
        user_vector = np.array([user_prefs.get(key, 0) for key in feature_keys])
        
        # Calculate similarity between user preferences and items
        similarities = cosine_similarity(
            user_vector.reshape(1, -1),
            self.item_features
        )[0]
        
        # Get top N items
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = [
            {
                'item_id': item_ids[idx],
                'score': float(similarities[idx]),
                'features': self.items[item_ids[idx]]
            }
            for idx in top_indices
        ]
        
        return recommendations
    
    def save_data(self, filename='recommendation_data.json'):
        """Save engine data to file"""
        data = {
            'users': self.users,
            'items': self.items
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Data saved to {filename}')
    
    def load_data(self, filename='recommendation_data.json'):
        """Load engine data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.users = data.get('users', {})
            self.items = data.get('items', {})
            print(f'Data loaded from {filename}')
        except FileNotFoundError:
            print(f'File {filename} not found')


# Example usage
if __name__ == '__main__':
    # Create recommendation engine
    engine = SmartRecEngine()
    
    # Add sample movies with features (genre scores)
    engine.add_item('movie1', {'action': 0.9, 'comedy': 0.1, 'drama': 0.3, 'scifi': 0.8})
    engine.add_item('movie2', {'action': 0.2, 'comedy': 0.9, 'drama': 0.1, 'scifi': 0.1})
    engine.add_item('movie3', {'action': 0.8, 'comedy': 0.2, 'drama': 0.5, 'scifi': 0.9})
    engine.add_item('movie4', {'action': 0.1, 'comedy': 0.8, 'drama': 0.9, 'scifi': 0.2})
    engine.add_item('movie5', {'action': 0.7, 'comedy': 0.3, 'drama': 0.6, 'scifi': 0.7})
    
    # Add users with preferences
    engine.add_user('user1', {'action': 0.9, 'comedy': 0.2, 'drama': 0.4, 'scifi': 0.8})
    engine.add_user('user2', {'action': 0.1, 'comedy': 0.9, 'drama': 0.8, 'scifi': 0.2})
    
    print('\n=== Content-Based Recommendations ===')
    print('\nMovies similar to movie1:')
    recommendations = engine.recommend_content_based('movie1', top_n=3)
    for rec in recommendations:
        print(f"  {rec['item_id']}: Score {rec['score']:.3f}")
    
    print('\n=== User-Based Recommendations ===')
    print('\nRecommendations for user1:')
    recommendations = engine.recommend_for_user('user1', top_n=3)
    for rec in recommendations:
        print(f"  {rec['item_id']}: Score {rec['score']:.3f}")
    
    print('\nRecommendations for user2:')
    recommendations = engine.recommend_for_user('user2', top_n=3)
    for rec in recommendations:
        print(f"  {rec['item_id']}: Score {rec['score']:.3f}")
    
    # Save data
    engine.save_data()
