import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from faker import Faker
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Initialize Faker
faker = Faker()
np.random.seed(42)

# Step 1: Generate Synthetic Dataset
n_websites = 10000
data = {
    'website_id': [f"WEB_{str(i).zfill(5)}" for i in range(1, n_websites + 1)],
    'low_quality_backlinks_ratio': [],
    'external_links_count': [],
    'domain_authority': [],
    'backlink_diversity': [],
    'keyword_stuffing_score': [],
    'content_length': [],
    'meta_description_missing': [],
    'duplicate_content_ratio': [],
    'page_load_time': [],
    'mobile_friendliness': [],
    'https_usage': [],
    'broken_links_ratio': [],
    'ads_to_content_ratio': [],
    'user_engagement_score': [],
    'site_age': [],
    'is_spammy': []
}

for _ in range(n_websites):
    is_spammy = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% spammy websites
    if is_spammy:
        # Spammy website patterns
        low_quality_backlinks_ratio = np.random.uniform(50, 90)  # High low-quality backlinks
        external_links_count = np.random.randint(50, 500)  # Excessive outbound links
        domain_authority = np.random.randint(1, 30)  # Low DA
        backlink_diversity = np.random.uniform(0.1, 0.4)  # Low diversity
        keyword_stuffing_score = np.random.uniform(30, 80)  # High keyword stuffing
        content_length = np.random.randint(100, 500)  # Thin content
        meta_description_missing = np.random.uniform(30, 80)  # Many missing meta descriptions
        duplicate_content_ratio = np.random.uniform(40, 90)  # High duplicate content
        page_load_time = np.random.uniform(5, 10)  # Slow load time
        mobile_friendliness = np.random.choice([0, 1], p=[0.7, 0.3])  # Often not mobile-friendly
        https_usage = np.random.choice([0, 1], p=[0.6, 0.4])  # Often HTTP
        broken_links_ratio = np.random.uniform(20, 60)  # Many broken links
        ads_to_content_ratio = np.random.uniform(40, 80)  # Ad-heavy
        user_engagement_score = np.random.uniform(10, 40)  # Poor engagement
        site_age = np.random.uniform(0, 2)  # Newer domains
    else:
        # High-quality website patterns
        low_quality_backlinks_ratio = np.random.uniform(5, 20)  # Few low-quality backlinks
        external_links_count = np.random.randint(5, 50)  # Moderate outbound links
        domain_authority = np.random.randint(40, 90)  # High DA
        backlink_diversity = np.random.uniform(0.6, 0.9)  # High diversity
        keyword_stuffing_score = np.random.uniform(0, 10)  # Low keyword stuffing
        content_length = np.random.randint(1000, 5000)  # Long content
        meta_description_missing = np.random.uniform(0, 10)  # Few missing meta descriptions
        duplicate_content_ratio = np.random.uniform(0, 15)  # Low duplicate content
        page_load_time = np.random.uniform(0.5, 3)  # Fast load time
        mobile_friendliness = np.random.choice([0, 1], p=[0.1, 0.9])  # Mostly mobile-friendly
        https_usage = np.random.choice([0, 1], p=[0.05, 0.95])  # Mostly HTTPS
        broken_links_ratio = np.random.uniform(0, 10)  # Few broken links
        ads_to_content_ratio = np.random.uniform(5, 20)  # Content-focused
        user_engagement_score = np.random.uniform(60, 90)  # Good engagement
        site_age = np.random.uniform(2, 15)  # Older domains
    
    data['low_quality_backlinks_ratio'].append(round(low_quality_backlinks_ratio, 2))
    data['external_links_count'].append(external_links_count)
    data['domain_authority'].append(domain_authority)
    data['backlink_diversity'].append(round(backlink_diversity, 2))
    data['keyword_stuffing_score'].append(round(keyword_stuffing_score, 2))
    data['content_length'].append(content_length)
    data['meta_description_missing'].append(round(meta_description_missing, 2))
    data['duplicate_content_ratio'].append(round(duplicate_content_ratio, 2))
    data['page_load_time'].append(round(page_load_time, 2))
    data['mobile_friendliness'].append(mobile_friendliness)
    data['https_usage'].append(https_usage)
    data['broken_links_ratio'].append(round(broken_links_ratio, 2))
    data['ads_to_content_ratio'].append(round(ads_to_content_ratio, 2))
    data['user_engagement_score'].append(round(user_engagement_score, 2))
    data['site_age'].append(round(site_age, 2))
    data['is_spammy'].append(is_spammy)

df = pd.DataFrame(data)
df.to_csv('synthetic_website_data.csv', index=False)
print(f"Generated synthetic dataset with {len(df)} websites.")

# Step 2: Data Preprocessing
features = [
    'low_quality_backlinks_ratio', 'external_links_count', 'domain_authority',
    'backlink_diversity', 'keyword_stuffing_score', 'content_length',
    'meta_description_missing', 'duplicate_content_ratio', 'page_load_time',
    'mobile_friendliness', 'https_usage', 'broken_links_ratio',
    'ads_to_content_ratio', 'user_engagement_score', 'site_age'
]
X = df[features]
y = df['is_spammy']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 3: Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance in Random Forest Classifier for Spam Score')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'.")

# Step 4: Predict Spam Score Probabilities
probs = model.predict_proba(X_scaled)[:, 1]
df['spam_score_prob'] = probs

# Top 20 websites with highest spam score probability
top_20_spammy = df[['website_id', 'spam_score_prob']].sort_values(
    'spam_score_prob', ascending=False
).head(20)
print("\nTop 20 Websites with Highest Spam Score Probability:")
print(top_20_spammy.round(4))

# Top 5 websites with spam score probability < 5% (high-quality)
top_5_high_quality = df[df['spam_score_prob'] < 0.05][
    ['website_id', 'spam_score_prob']
].sort_values('spam_score_prob', ascending=True).head(5)
print("\nTop 5 Websites with Spam Score < 5% (High-Quality for Google Rankings):")
print(top_5_high_quality.round(4))

# Save results
top_20_spammy.to_csv('top_20_spammy_websites.csv', index=False)
top_5_high_quality.to_csv('top_5_high_quality_websites.csv', index=False)
print("\nResults saved to 'top_20_spammy_websites.csv' and 'top_5_high_quality_websites.csv'.")