import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def clean_text(text):
    """Small helper to clean text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove @mentions and #
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.strip()

def map_sentiment(sentiment):
    """Group 200+ sentiments into 5 core labels for better ML performance."""
    s = str(sentiment).strip().lower()
    
    positive_keywords = ['positive', 'joy', 'happy', 'happiness', 'excitement', 'elation', 'euphoria', 
                        'contentment', 'serenity', 'gratitude', 'awe', 'hope', 'proud', 'love', 
                        'amusement', 'enjoyment', 'admiration', 'affection', 'adoration', 'kind', 
                        'harmony', 'fulfillment', 'reverence', 'compassion', 'warmth', 'ecstasy']
    
    negative_keywords = ['negative', 'anger', 'fear', 'sadness', 'disgust', 'disappointed', 'bitter', 
                        'shame', 'despair', 'grief', 'loneliness', 'sad', 'hate', 'bad', 'frustrated',
                        'regret', 'betrayal', 'suffering', 'melancholy', 'devastated', 'heartbreak']
    
    neutral_keywords = ['neutral', 'calmness', 'confusion', 'anticipation', 'surprise', 'acceptance',
                       'indifference', 'curiosity', 'numbness', 'nostalgia', 'contemplation', 'ambivalence']

    for word in positive_keywords:
        if word in s: return 'Positive'
    for word in negative_keywords:
        if word in s: return 'Negative'
    for word in neutral_keywords:
        if word in s: return 'Neutral'
    
    return 'Other'

def comprehensive_preprocess(file_path):
    print(f"--- Loading dataset from {file_path} ---")
    df = pd.read_csv(file_path)

    # 1. CLEANING & UNIFICATION
    print("Step 1: Unifying inconsistent categories (stripping spaces, standardizing)...")
    categorical_cols = ['Platform', 'Country']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # 2. SENTIMENT CONSOLIDATION
    print("Step 2: Consolidating 279 sentiments into 4 core classes...")
    df['Sentiment_Group'] = df['Sentiment'].apply(map_sentiment)
    print(f"Resulting Groups:\n{df['Sentiment_Group'].value_counts()}\n")

    # 3. TEXT PROCESSING (TF-IDF)
    print("Step 3: Processing 'Text' content with TF-IDF Vectorization...")
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    
    # We'll use the top 100 words as features for this example
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Cleaned_Text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    
    # 4. HASHTAGS EXTRACTION
    print("Step 4: Extracting counts from 'Hashtags'...")
    # Count the number of tags to use as a feature
    df['Hashtag_Count'] = df['Hashtags'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

    # 5. MISSING VALUES check
    print("Step 5: Final check for missing numeric data...")
    df['Retweets'] = df['Retweets'].fillna(0)
    df['Likes'] = df['Likes'].fillna(0)

    # 6. ENCODING CATEGORIES
    print("Step 6: Encoding categorical variables...")
    le_target = LabelEncoder()
    df['Target'] = le_target.fit_transform(df['Sentiment_Group'])
    
    le_platform = LabelEncoder()
    df['Platform_Enc'] = le_platform.fit_transform(df['Platform'])
    
    le_country = LabelEncoder()
    df['Country_Enc'] = le_country.fit_transform(df['Country'])

    # 7. SCALING
    print("Step 7: Scaling numerical features...")
    scaler = StandardScaler()
    numeric_features = ['Retweets', 'Likes', 'Year', 'Month', 'Day', 'Hour', 'Hashtag_Count']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 8. CONSOLIDATING FINAL FEATURE SET
    # Concatenate the TF-IDF features with our metadata features
    X = pd.concat([df[numeric_features + ['Platform_Enc', 'Country_Enc']], tfidf_df], axis=1)
    y = df['Target']

    # 9. SPLITTING
    print("Step 8: Splitting dataset into training (80%) and testing (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Summary ---")
    print(f"Original unique sentiments: {df['Sentiment'].nunique()}")
    print(f"Final target classes: {le_target.classes_}")
    print(f"Final feature count: {X.shape[1]}")
    print(f"Final samples: {X.shape[0]}")

    # Save to CSV
    X_train.to_csv('X_train_v2.csv', index=False)
    X_test.to_csv('X_test_v2.csv', index=False)
    y_train.to_csv('y_train_v2.csv', index=False)
    y_test.to_csv('y_test_v2.csv', index=False)
    print("\nEnhanced files saved: X_train_v2, X_test_v2, etc.")

if __name__ == "__main__":
    file_path = "c:/Users/danie/OneDrive/Documents/DPM/3) Sentiment dataset (1).csv"
    if os.path.exists(file_path):
        comprehensive_preprocess(file_path)
    else:
        print(f"Error: File not found at {file_path}")
