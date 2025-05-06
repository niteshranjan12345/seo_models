import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prophet import Prophet
from faker import Faker
import uuid
import plotly.express as px
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('vader_lexicon')

# Initialize Faker
faker = Faker()

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Market Research
# Understand local SEO factors: keywords, reviews, proximity.
print("Market Research: Identified key factors - local keywords, search volume, review sentiment, proximity.")

# Step 2: Data Collection
# Generate synthetic dataset for 5,000 local searches.
n_searches = 5000
cities = ['Goa', 'Mumbai', 'Delhi', 'Jaipur', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Kerala', 'Udaipur']
landmarks = ['beach', 'fort', 'temple', 'market', 'park', 'museum', 'street', 'lake', 'palace', 'hill']
city_coords = {
    'Goa': (15.2993, 74.1240), 'Mumbai': (19.0760, 72.8777), 'Delhi': (28.7041, 77.1025),
    'Jaipur': (26.9124, 75.7873), 'Bangalore': (12.9716, 77.5946), 'Kolkata': (22.5726, 88.3639),
    'Chennai': (13.0827, 80.2707), 'Hyderabad': (17.3850, 78.4867), 'Kerala': (10.8505, 76.2711),
    'Udaipur': (24.5854, 73.7125)
}

def generate_keyword(city):
    landmark = np.random.choice(landmarks)
    return f"hotels near {city} {landmark}"

data = {
    'search_id': [str(uuid.uuid4()) for _ in range(n_searches)],
    'keyword': [],
    'location': np.random.choice(cities, n_searches),
    'latitude': [],
    'longitude': [],
    'search_volume': np.random.randint(100, 100000, n_searches),
    'review_count': np.random.randint(0, 500, n_searches),
    'avg_rating': np.random.uniform(1, 5, n_searches).round(1),
    'click_through_rate': np.random.uniform(0, 20, n_searches).round(2)  # 0–20%
}

# Generate keywords and coordinates
for i in range(n_searches):
    city = data['location'][i]
    data['keyword'].append(generate_keyword(city))
    lat, lon = city_coords[city]
    # Add noise to coordinates for realism
    data['latitude'].append(lat + np.random.normal(0, 0.05))
    data['longitude'].append(lon + np.random.normal(0, 0.05))

df = pd.DataFrame(data)

# Generate synthetic reviews and compute sentiment
sia = SentimentIntensityAnalyzer()
def generate_reviews():
    reviews = [faker.sentence() for _ in range(np.random.randint(1, 10))]
    return ' '.join(reviews)

df['reviews'] = [generate_reviews() for _ in range(n_searches)]
df['sentiment_score'] = df['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Generate search trend (12-month synthetic data)
def generate_trend():
    return np.random.uniform(0.8, 1.2, 12).tolist()  # Random fluctuations

df['search_trend'] = [generate_trend() for _ in range(n_searches)]

# Generate keyword impact score (target)
df['keyword_impact_score'] = (
    0.3 * (df['search_volume'] / 100000 * 100) +
    0.25 * (df['click_through_rate'] / 20 * 100) +
    0.2 * (df['avg_rating'] / 5 * 100) +
    0.15 * ((df['sentiment_score'] + 1) / 2 * 100) +
    0.1 * (df['review_count'] / 500 * 100)
).clip(lower=0, upper=100).round(2)

# Add noise
df['keyword_impact_score'] += np.random.normal(0, 5, n_searches).clip(min=-10, max=10)
df['keyword_impact_score'] = df['keyword_impact_score'].clip(lower=0, upper=100).round(2)

# Save synthetic dataset
df.to_csv('local_search_data.csv', index=False)
print("Data Collection: Generated synthetic dataset for 5,000 local searches.")

# Step 3: Data Cleaning
# Handle missing values and outliers.
print("Data Cleaning: Checking for missing values...")
print(df.isnull().sum())

# Cap outliers
df['search_volume'] = df['search_volume'].clip(lower=100, upper=100000)
df['click_through_rate'] = df['click_through_rate'].clip(lower=0, upper=20)
df['avg_rating'] = df['avg_rating'].clip(lower=1, upper=5)
df['review_count'] = df['review_count'].clip(lower=0, upper=500)
df['sentiment_score'] = df['sentiment_score'].clip(lower=-1, upper=1)

print("Data Cleaning: Handled outliers and ensured valid ranges.")

# Step 4: Exploratory Data Analysis (EDA)
# Analyze search trends and sentiment.
print("EDA: Summary statistics...")
print(df.describe())

# Visualize sentiment by city
fig_eda = px.box(
    df,
    x='location',
    y='sentiment_score',
    title='Review Sentiment by City',
    labels={'location': 'City', 'sentiment_score': 'Sentiment Score (-1 to 1)'}
)
fig_eda.update_layout(template='plotly_white')
fig_eda.write_html('eda_sentiment_by_city.html')
print("EDA: Generated box plot of sentiment by city.")

# Step 5: Feature Engineering
# Create geospatial features and encode categorical variables.
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])]
)

features = [
    'search_volume', 'click_through_rate', 'avg_rating', 'sentiment_score',
    'review_count', 'latitude', 'longitude'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

print("Feature Engineering: Created geospatial features and scaled numerical features.")

# Step 6: Geospatial Clustering
# Cluster locations by search patterns.
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
fig_clusters = px.scatter_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    color='cluster',
    hover_data=['keyword', 'search_volume', 'location'],
    title='Geospatial Clusters of Local Search Trends',
    mapbox_style='open-street-map',
    zoom=4
)
fig_clusters.update_layout(template='plotly_white')
fig_clusters.write_html('geospatial_clusters.html')
print("Geospatial Clustering: Clustered searches and generated map.")

# Step 7: Time-Series Forecasting
# Forecast search volume for top keywords.
def forecast_search_volume(keyword_data):
    trend = pd.DataFrame({
        'ds': pd.date_range(start='2024-04-01', periods=12, freq='M'),
        'y': keyword_data['search_trend']
    })
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(trend)
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(6).to_dict()

# Forecast for top 10 keywords (to be identified later)
print("Time-Series Forecasting: Prepared forecasting for top keywords.")

# Step 8: Model Selection
# Choose Random Forest Regressor for keyword impact prediction.
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Model Selection: Chose Random Forest Regressor.")

# Step 9: Model Training
# Split data and train.
X = X_scaled
y = df['keyword_impact_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model Training: Trained Random Forest Regressor.")

# Step 10: Model Evaluation
# Evaluate with MAE, RMSE, R².
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation: Performance metrics...")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 11: Prediction and Insights
# Predict keyword impact scores and identify top 10 high-impact keywords.
df['predicted_impact_score'] = model.predict(X_scaled)

# Top 10 high-impact keywords
top_10_keywords = df[['search_id', 'keyword', 'location', 'search_volume', 'click_through_rate', 'avg_rating', 'sentiment_score', 'predicted_impact_score']].sort_values(by='predicted_impact_score', ascending=False).head(10)

# Save predictions
df.to_csv('search_predictions.csv', index=False)
top_10_keywords.to_csv('top_10_high_impact_keywords.csv', index=False)

print("\nTop 10 High-Impact Keywords for Local SEO:")
print(top_10_keywords[['search_id', 'keyword', 'location', 'predicted_impact_score']])

# Generate GMB optimization recommendations
recommendations = []
for city in cities:
    city_keywords = top_10_keywords[top_10_keywords['location'] == city]['keyword'].tolist()
    if city_keywords:
        primary_keyword = city_keywords[0]
    else:
        primary_keyword = f"hotels in {city}"
    rec = {
        'city': city,
        'gmb_title': f"{city} Luxury Hotels",
        'gmb_description': f"Book top-rated hotels in {city} near popular landmarks like {primary_keyword.split('near')[-1].strip()}. Enjoy premium amenities and exceptional service.",
        'gmb_categories': 'Hotel, Resort, Lodging',
        'keywords': ', '.join(city_keywords[:3]) if city_keywords else primary_keyword
    }
    recommendations.append(rec)

rec_df = pd.DataFrame(recommendations)
rec_df.to_csv('gmb_optimization_recommendations.csv', index=False)
print("\nGMB Optimization Recommendations:")
print(rec_df)

# Forecast search volume for top keyword
top_keyword_data = top_10_keywords.iloc[0]
forecast = forecast_search_volume(top_keyword_data)
forecast_df = pd.DataFrame(forecast)
fig_forecast = px.line(
    forecast_df,
    x='ds',
    y='yhat',
    title=f"Search Volume Forecast for '{top_keyword_data['keyword']}'",
    labels={'ds': 'Date', 'yhat': 'Predicted Search Volume'}
)
fig_forecast.update_layout(template='plotly_white')
fig_forecast.write_html('search_volume_forecast.html')
print("Time-Series Forecasting: Generated forecast plot for top keyword.")

# Step 12: Visualization
# Scatter plot: Predicted vs. actual impact scores
fig_pred = px.scatter(
    x=y_test, y=y_pred,
    title='Predicted vs. Actual Keyword Impact Scores',
    labels={'x': 'Actual Impact Score', 'y': 'Predicted Impact Score'}
)
fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='red', dash='dash'))
fig_pred.update_layout(template='plotly_white')
fig_pred.write_html('predicted_vs_actual_scores.html')
print("Visualization: Generated scatter, geospatial, and forecast plots.")

# Step 13: Deployment
# Simulate Flask API for keyword scoring.
app = Flask(__name__)

@app.route('/score_keyword', methods=['POST'])
def score_keyword():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df['sentiment_score'] = input_df['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
    input_features = input_df[features]
    input_scaled = scaler.transform(input_features)
    impact_score = model.predict(input_scaled)[0]
    return jsonify({'predicted_impact_score': round(impact_score, 2)})

# Simulate running Flask app (commented out)
# if __name__ == '__main__':
#     app.run(debug=True)
print("Deployment: Simulated Flask API for keyword impact scoring.")

# Step 14: Monitoring and Maintenance
# Plan to monitor and update model.
print("Monitoring and Maintenance: Monitor search trends, collect new GMB data, and retrain periodically.")