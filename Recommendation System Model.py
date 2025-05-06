import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from faker import Faker
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Initialize Faker
faker = Faker('en_IN')
np.random.seed(42)

# Step 1: Generate Synthetic Dataset
n_movies = 20000
directors = ['Karan Johar', 'Sanjay Leela Bhansali', 'Anurag Kashyap', 'Rohit Shetty', 
             'Aditya Chopra', 'Zoya Akhtar', 'Rajkumar Hirani', 'Imtiaz Ali', 
             'Nitesh Tiwari', 'Shoojit Sircar']
actors = ['Shah Rukh Khan', 'Deepika Padukone', 'Ranbir Kapoor', 'Priyanka Chopra', 
          'Salman Khan', 'Aamir Khan', 'Kangana Ranaut', 'Ranveer Singh', 
          'Alia Bhatt', 'Akshay Kumar', 'Hrithik Roshan', 'Kareena Kapoor']
production_houses = ['Yash Raj Films', 'Dharma Productions', 'Eros International', 
                     'T-Series', 'Red Chillies Entertainment', 'Reliance Entertainment']
languages = ['Hindi', 'Tamil', 'Telugu', 'Marathi', 'Punjabi']

data = {
    'movie_id': [f"MOV_{str(i).zfill(5)}" for i in range(1, n_movies + 1)],
    'title': [],
    'release_year': [],
    'runtime': [],
    'director': [],
    'actors': [],
    'imdb_rating': [],
    'budget': [],
    'box_office': [],
    'language': [],
    'production_house': [],
    'sentiment_score': [],
    'scare_factor': [],
    'action_intensity': [],
    'is_romantic': [],
    'is_horror': [],
    'is_action': [],
    'is_drama': []
}

for _ in range(n_movies):
    # Basic metadata
    title = f"{faker.word().capitalize()} {np.random.choice(['Dil', 'Pyar', 'Dhadkan', 'Ishq', 'Veer', 'Shakti'])}"
    release_year = np.random.randint(2000, 2026)
    runtime = np.random.randint(90, 240)
    director = np.random.choice(directors)
    actors_list = ', '.join(np.random.choice(actors, size=np.random.randint(2, 5), replace=False))
    language = np.random.choice(languages, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    production_house = np.random.choice(production_houses)
    
    # Financial and rating
    budget = np.random.randint(1, 500)
    imdb_rating = np.random.uniform(3.0, 9.5)
    box_office = int(budget * np.random.uniform(0, 4) * (imdb_rating / 7))  # Correlated with rating
    
    # Genre-specific features
    genres = []
    if np.random.random() < 0.4: genres.append('romantic')
    if np.random.random() < 0.1: genres.append('horror')
    if np.random.random() < 0.3: genres.append('action')
    if np.random.random() < 0.5: genres.append('drama')
    if not genres: genres.append(np.random.choice(['comedy', 'thriller']))
    
    # Feature patterns based on genres
    if 'romantic' in genres:
        sentiment_score = np.random.uniform(0.7, 1.0)
        scare_factor = np.random.uniform(0.0, 0.2)
        action_intensity = np.random.uniform(0.0, 0.3)
    elif 'horror' in genres:
        sentiment_score = np.random.uniform(0.0, 0.3)
        scare_factor = np.random.uniform(0.7, 1.0)
        action_intensity = np.random.uniform(0.0, 0.3)
    elif 'action' in genres:
        sentiment_score = np.random.uniform(0.3, 0.6)
        scare_factor = np.random.uniform(0.0, 0.2)
        action_intensity = np.random.uniform(0.7, 1.0)
    else:  # Drama or others
        sentiment_score = np.random.uniform(0.5, 0.8)
        scare_factor = np.random.uniform(0.0, 0.2)
        action_intensity = np.random.uniform(0.0, 0.4)
    
    data['title'].append(title)
    data['release_year'].append(release_year)
    data['runtime'].append(runtime)
    data['director'].append(director)
    data['actors'].append(actors_list)
    data['imdb_rating'].append(round(imdb_rating, 1))
    data['budget'].append(budget)
    data['box_office'].append(box_office)
    data['language'].append(language)
    data['production_house'].append(production_house)
    data['sentiment_score'].append(round(sentiment_score, 2))
    data['scare_factor'].append(round(scare_factor, 2))
    data['action_intensity'].append(round(action_intensity, 2))
    data['is_romantic'].append(1 if 'romantic' in genres else 0)
    data['is_horror'].append(1 if 'horror' in genres else 0)
    data['is_action'].append(1 if 'action' in genres else 0)
    data['is_drama'].append(1 if 'drama' in genres else 0)

df = pd.DataFrame(data)
df.to_csv('bollywood_movies.csv', index=False)
print(f"Generated synthetic dataset with {len(df)} Bollywood movies.")

# Step 2: Data Preprocessing
# Encode categorical features
le_director = LabelEncoder()
le_actors = LabelEncoder()
le_language = LabelEncoder()
le_production = LabelEncoder()

df['director'] = le_director.fit_transform(df['director'])
df['actors'] = le_actors.fit_transform(df['actors'])
df['language'] = le_language.fit_transform(df['language'])
df['production_house'] = le_production.fit_transform(df['production_house'])

# Features and targets
features = [
    'release_year', 'runtime', 'director', 'actors', 'imdb_rating',
    'budget', 'box_office', 'language', 'production_house',
    'sentiment_score', 'scare_factor', 'action_intensity'
]
X = df[features]
y = df[['is_romantic', 'is_horror', 'is_action', 'is_drama']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train Random Forest Classifier for Multi-label Classification
# Train separate classifiers for each genre
classifiers = {}
probs = {}
for genre in ['is_romantic', 'is_horror', 'is_action', 'is_drama']:
    clf = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y[genre], test_size=0.2, random_state=42
    )
    clf.fit(X_train, y_train)
    classifiers[genre] = clf
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"\nModel Performance for {genre}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Predict probabilities
    probs[genre] = clf.predict_proba(X_scaled)[:, 1]

# Feature importance (using romantic classifier as example)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': classifiers['is_romantic'].feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance (Romantic Classifier):")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance in Random Forest Classifier (Romantic)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'.")

# Step 4: Movie Recommendations
df['romantic_prob'] = probs['is_romantic']
df['horror_prob'] = probs['is_horror']
df['action_prob'] = probs['is_action']
df['drama_prob'] = probs['is_drama']

# Top 12 Romantic Movies
top_12_romantic = df[df['imdb_rating'] > 7.0][
    ['movie_id', 'title', 'imdb_rating', 'romantic_prob']
].sort_values('romantic_prob', ascending=False).head(12)
print("\nTop 12 Romantic Movies:")
print(top_12_romantic.round(4))

# Top 10 Horror Movies
top_10_horror = df[df['imdb_rating'] > 7.0][
    ['movie_id', 'title', 'imdb_rating', 'horror_prob']
].sort_values('horror_prob', ascending=False).head(10)
print("\nTop 10 Horror Movies:")
print(top_10_horror.round(4))

# Top 5 Action and Drama Movies
df['action_drama_prob'] = (df['action_prob'] + df['drama_prob']) / 2
top_5_action_drama = df[(df['imdb_rating'] > 7.0) & (df['is_action'] == 1) & (df['is_drama'] == 1)][
    ['movie_id', 'title', 'imdb_rating', 'action_drama_prob']
].sort_values('action_drama_prob', ascending=False).head(5)
print("\nTop 5 Action and Drama Movies:")
print(top_5_action_drama.round(4))

# Save results
top_12_romantic.to_csv('top_12_romantic_movies.csv', index=False)
top_10_horror.to_csv('top_10_horror_movies.csv', index=False)
top_5_action_drama.to_csv('top_5_action_drama_movies.csv', index=False)
print("\nResults saved to 'top_12_romantic_movies.csv', 'top_10_horror_movies.csv', and 'top_5_action_drama_movies.csv'.")