import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def preprocess_posts():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    input_path = os.path.join(data_dir, "discourse_posts.json")
    output_path = os.path.join(data_dir, "post_index.pkl")
    
    # Load posts
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception as e:
        print(f"Error loading posts: {e}")
        return
    
    # Extract texts and metadata
    documents = [post["text"] for post in posts]
    metadata = [{"url": post["url"], "text": post["text"]} for post in posts]
    
    # Create TF-IDF index
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Save index and metadata
    index_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "metadata": metadata
    }
    try:
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)
        print(f"Saved index to {output_path}")
    except Exception as e:
        print(f"Error saving index: {e}")

if __name__ == "__main__":
    preprocess_posts()