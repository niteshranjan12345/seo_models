import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
import spacy
from faker import Faker
import uuid
import plotly.express as px
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('vader_lexicon')

# Initialize spaCy and Faker
nlp = spacy.load('en_core_web_sm')
faker = Faker()

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Market Research
# Understand content quality factors: readability, keyword density, sentiment, engagement.
print("Market Research: Identified key features - readability, keyword density, sentiment, unique words, links, engagement metrics.")

# Step 2: Data Collection
# Generate synthetic dataset for 5,000 hotel content pieces.
n_content = 5000
destinations = ['Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Kerala', 'Udaipur']
content_types = ['Blog Post', 'Landing Page', 'Hotel Description', 'Travel Guide']
keywords = ['hotel', 'luxury', 'booking', 'travel', 'vacation', 'stay', 'resort', 'destination']

def generate_content():
    destination = np.random.choice(destinations)
    content_type = np.random.choice(content_types)
    title = f"{content_type}: {faker.sentence(nb_words=5)} in {destination}"
    word_count = np.random.randint(300, 2000)
    sentences = [faker.sentence() for _ in range(np.random.randint(10, 50))]
    # Inject keywords
    for keyword in np.random.choice(keywords, size=np.random.randint(5, 20)):
        sentences[np.random.randint(0, len(sentences))] += f" {keyword}."
    text = ' '.join(sentences)
    return title, text, word_count

data = {
    'content_id': [str(uuid.uuid4()) for _ in range(n_content)],
    'content_title': [],
    'content_text': [],
    'word_count': [],
    'keyword_density': np.random.uniform(1, 5, n_content).round(2),  # 1–5%
    'internal_links': np.random.randint(0, 20, n_content),
    'external_links': np.random.randint(0, 10, n_content),
    'bounce_rate': np.random.uniform(20, 80, n_content).round(2),  # 20–80%
    'time_on_page': np.random.uniform(30, 600, n_content).round(2)  # 30–600 seconds
}

# Generate content and compute word count
for _ in range(n_content):
    title, text, word_count = generate_content()
    data['content_title'].append(title)
    data['content_text'].append(text)
    data['word_count'].append(word_count)

df = pd.DataFrame(data)

# Compute NLP-based features
sia = SentimentIntensityAnalyzer()
df['readability_score'] = df['content_text'].apply(flesch_reading_ease).clip(lower=0, upper=100)
df['sentiment_score'] = df['content_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Compute unique words ratio
def unique_words_ratio(text):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    return len(set(words)) / len(words) if words else 0

df['unique_words_ratio'] = df['content_text'].apply(unique_words_ratio).round(2)

# Generate synthetic content quality score (target)
df['content_quality_score'] = (
    0.2 * df['readability_score'] +
    0.2 * (100 - df['bounce_rate']) +
    0.2 * (df['time_on_page'] / 600 * 100) +
    0.15 * df['sentiment_score'] * 50 +
    0.1 * (df['keyword_density'] / 5 * 100) +
    0.1 * (df['unique_words_ratio'] * 100) +
    0.05 * (df['internal_links'] / 20 * 100) +
    0.05 * (df['external_links'] / 10 * 100)
).clip(lower=0, upper=100).round(2)

# Add noise to simulate real-world variability (Fixed Bug: Use min/max instead of lower/upper)
df['content_quality_score'] += np.random.normal(0, 5, n_content).clip(min=-10, max=10)
df['content_quality_score'] = df['content_quality_score'].clip(lower=0, upper=100).round(2)

# Save synthetic dataset
df.to_csv('hotel_content_quality.csv', index=False)
print("Data Collection: Generated synthetic dataset for 5,000 hotel content pieces.")

# Step 3: Data Cleaning
# Handle missing values and outliers.
print("Data Cleaning: Checking for missing values...")
print(df.isnull().sum())

# Cap outliers
df['keyword_density'] = df['keyword_density'].clip(lower=1, upper=5)
df['bounce_rate'] = df['bounce_rate'].clip(lower=20, upper=80)
df['time_on_page'] = df['time_on_page'].clip(lower=30, upper=600)
df['readability_score'] = df['readability_score'].clip(lower=0, upper=100)
df['unique_words_ratio'] = df['unique_words_ratio'].clip(lower=0, upper=1)
df['internal_links'] = df['internal_links'].clip(lower=0, upper=20)
df['external_links'] = df['external_links'].clip(lower=0, upper=10)

print("Data Cleaning: Handled outliers and ensured valid ranges.")

# Step 4: Exploratory Data Analysis (EDA)
# Analyze feature distributions.
print("EDA: Summary statistics...")
print(df.describe())

# Visualize content quality score distribution
fig_eda = px.histogram(
    df,
    x='content_quality_score',
    title='Distribution of Content Quality Scores',
    labels={'content_quality_score': 'Content Quality Score (0–100)'},
    nbins=50
)
fig_eda.update_layout(template='plotly_white')
fig_eda.write_html('eda_content_quality_distribution.html')
print("EDA: Generated histogram of content quality scores.")

# Step 5: Feature Engineering
# Select features for modeling and scale numerical features.
features = [
    'word_count', 'keyword_density', 'readability_score', 'sentiment_score',
    'unique_words_ratio', 'internal_links', 'external_links',
    'bounce_rate', 'time_on_page'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

print("Feature Engineering: Selected and scaled numerical features.")

# Step 6: Model Selection
# Choose Random Forest Regressor for content quality prediction.
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Model Selection: Chose Random Forest Regressor.")

# Step 7: Model Training
# Split data and train.
X = X_scaled
y = df['content_quality_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model Training: Trained Random Forest Regressor.")

# Step 8: Model Evaluation
# Evaluate with MAE, RMSE, R².
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation: Performance metrics...")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Model Tuning
# Grid search for hyperparameter optimization.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Model Tuning: Best parameters:", grid_search.best_params_)

# Re-evaluate with best model
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print("Model Evaluation (Tuned): Performance metrics...")
print(f"Mean Absolute Error (MAE): {mae_best:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_best:.2f}")
print(f"R² Score: {r2_best:.2f}")

# Step 10: Prediction and Insights
# Predict content quality scores and identify top 10 high-quality and top 5 low-quality content.
df['predicted_quality_score'] = best_model.predict(X_scaled)

# Top 10 high-quality content
top_10_high_quality = df[['content_id', 'content_title', 'word_count', 'keyword_density', 'readability_score', 'sentiment_score', 'bounce_rate', 'time_on_page', 'predicted_quality_score']].sort_values(by='predicted_quality_score', ascending=False).head(10)

# Top 5 low-quality content
top_5_low_quality = df[['content_id', 'content_title', 'word_count', 'keyword_density', 'readability_score', 'sentiment_score', 'bounce_rate', 'time_on_page', 'predicted_quality_score']].sort_values(by='predicted_quality_score').head(5)

# Save outputs
df.to_csv('content_quality_predictions.csv', index=False)
top_10_high_quality.to_csv('top_10_high_quality_content.csv', index=False)
top_5_low_quality.to_csv('top_5_low_quality_content.csv', index=False)

print("\nTop 10 Content Pieces with Highest Quality Scores:")
print(top_10_high_quality[['content_id', 'content_title', 'predicted_quality_score']])

print("\nTop 5 Content Pieces with Lowest Quality Scores:")
print(top_5_low_quality[['content_id', 'content_title', 'predicted_quality_score']])

# Step 11: Visualization
# Scatter plot: Predicted vs. actual quality scores
fig_pred = px.scatter(
    x=y_test, y=y_pred_best,
    title='Predicted vs. Actual Content Quality Scores',
    labels={'x': 'Actual Quality Score', 'y': 'Predicted Quality Score'}
)
fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='red', dash='dash'))
fig_pred.update_layout(template='plotly_white')
fig_pred.write_html('predicted_vs_actual_scores.html')

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

fig_importance = px.bar(
    feature_importance,
    x='importance',
    y='feature',
    title='Feature Importance in Random Forest Regressor',
    labels={'importance': 'Importance', 'feature': 'Feature'},
    orientation='h'
)
fig_importance.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
fig_importance.write_html('feature_importance.html')
print("Visualization: Generated scatter and feature importance plots.")

# Step 12: Deployment
# Simulate Flask API for content quality scoring.
app = Flask(__name__)

@app.route('/score_content', methods=['POST'])
def score_content():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df['readability_score'] = input_df['content_text'].apply(flesch_reading_ease).clip(lower=0, upper=100)
    input_df['sentiment_score'] = input_df['content_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    input_df['unique_words_ratio'] = input_df['content_text'].apply(unique_words_ratio)
    input_features = input_df[features]
    input_scaled = scaler.transform(input_features)
    quality_score = best_model.predict(input_scaled)[0]
    return jsonify({'predicted_quality_score': round(quality_score, 2)})

# Simulate running Flask app (commented out)
# if __name__ == '__main__':
#     app.run(debug=True)
print("Deployment: Simulated Flask API for content quality scoring.")

# Step 13: Monitoring and Maintenance
# Plan to monitor and update model.
print("Monitoring and Maintenance: Monitor content performance, collect new engagement data, and retrain periodically.")