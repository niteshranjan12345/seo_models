import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from faker import Faker
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
faker = Faker('en_IN')

# Step 1: Generate Synthetic Dataset
n_websites = 5000
destinations = ['Mumbai', 'Delhi', 'Goa', 'Jaipur', 'Kerala', 'Shimla', 'Udaipur', 
                'Bangalore', 'Chennai', 'Hyderabad']
amenities = ['spa', 'infinity pool', 'beach view', 'mountain view', 'free breakfast']

data = {
    'website_id': [f"WEB_{str(i).zfill(5)}" for i in range(1, n_websites + 1)],
    'description': [],
    'lcp': [],
    'fid': [],
    'cls': [],
    'page_size': [],
    'num_requests': [],
    'image_optimization': [],
    'server_response_time': [],
    'js_execution_time': [],
    'mobile_friendliness': [],
    'https_usage': [],
    'content_length': [],
    'keyword_density': [],
    'cwv_score': []
}

for _ in range(n_websites):
    # Performance category: Good (30%), Medium (40%), Poor (30%)
    performance = np.random.choice(['good', 'medium', 'poor'], p=[0.3, 0.4, 0.3])
    
    # Generate description
    city = np.random.choice(destinations)
    amenity = np.random.choice(amenities)
    description = f"Luxury hotel in {city} with {amenity} and premium services"
    
    # Features based on performance
    if performance == 'good':
        lcp = np.random.uniform(1, 2.5)
        fid = np.random.uniform(50, 100)
        cls = np.random.uniform(0, 0.1)
        page_size = np.random.uniform(0.5, 2)
        num_requests = np.random.randint(10, 50)
        image_optimization = np.random.uniform(80, 100)
        server_response_time = np.random.uniform(50, 200)
        js_execution_time = np.random.uniform(100, 500)
        mobile_friendliness = np.random.choice([0, 1], p=[0.1, 0.9])
        https_usage = np.random.choice([0, 1], p=[0.05, 0.95])
        content_length = np.random.randint(1000, 5000)
        keyword_density = np.random.uniform(1, 2)
    elif performance == 'medium':
        lcp = np.random.uniform(2.5, 4)
        fid = np.random.uniform(100, 300)
        cls = np.random.uniform(0.1, 0.25)
        page_size = np.random.uniform(2, 5)
        num_requests = np.random.randint(50, 100)
        image_optimization = np.random.uniform(50, 80)
        server_response_time = np.random.uniform(200, 500)
        js_execution_time = np.random.uniform(500, 1000)
        mobile_friendliness = np.random.choice([0, 1], p=[0.3, 0.7])
        https_usage = np.random.choice([0, 1], p=[0.2, 0.8])
        content_length = np.random.randint(500, 2000)
        keyword_density = np.random.uniform(2, 3)
    else:  # poor
        lcp = np.random.uniform(4, 10)
        fid = np.random.uniform(300, 1000)
        cls = np.random.uniform(0.25, 1)
        page_size = np.random.uniform(5, 10)
        num_requests = np.random.randint(100, 200)
        image_optimization = np.random.uniform(0, 50)
        server_response_time = np.random.uniform(500, 1000)
        js_execution_time = np.random.uniform(1000, 2000)
        mobile_friendliness = np.random.choice([0, 1], p=[0.6, 0.4])
        https_usage = np.random.choice([0, 1], p=[0.4, 0.6])
        content_length = np.random.randint(100, 1000)
        keyword_density = np.random.uniform(3, 5)
    
    # Calculate CWV score
    s_lcp = max(0, 100 - 25 * (lcp - 2.5)) if lcp <= 4 else 0
    s_fid = max(0, 100 - 0.333 * (fid - 100)) if fid <= 300 else 0
    s_cls = max(0, 100 - 400 * (cls - 0.1)) if cls <= 0.25 else 0
    cwv_score = 0.4 * s_lcp + 0.3 * s_fid + 0.3 * s_cls
    
    data['description'].append(description)
    data['lcp'].append(round(lcp, 2))
    data['fid'].append(round(fid, 2))
    data['cls'].append(round(cls, 2))
    data['page_size'].append(round(page_size, 2))
    data['num_requests'].append(num_requests)
    data['image_optimization'].append(round(image_optimization, 2))
    data['server_response_time'].append(round(server_response_time, 2))
    data['js_execution_time'].append(round(js_execution_time, 2))
    data['mobile_friendliness'].append(mobile_friendliness)
    data['https_usage'].append(https_usage)
    data['content_length'].append(content_length)
    data['keyword_density'].append(round(keyword_density, 2))
    data['cwv_score'].append(round(cwv_score, 2))

df = pd.DataFrame(data)
df.to_csv('hotel_cwv_dataset.csv', index=False)
print(f"Generated synthetic dataset with {len(df)} hotel websites.")

# Step 2: Data Preprocessing
# BERT embeddings for descriptions
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
bert_model.eval()

def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors='pt', max_length=32, 
            truncation=True, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

description_embeddings = get_bert_embeddings(df['description'].tolist())
print("Generated BERT embeddings for website descriptions.")

# Numerical features
num_features = [
    'lcp', 'fid', 'cls', 'page_size', 'num_requests', 'image_optimization',
    'server_response_time', 'js_execution_time', 'mobile_friendliness',
    'https_usage', 'content_length', 'keyword_density'
]
scaler = StandardScaler()
num_data = scaler.fit_transform(df[num_features])

# Combine BERT embeddings and numerical features
X = np.hstack([description_embeddings, num_data])
y = df['cwv_score'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Define BERT-based Regression Model
class BertRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(BertRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
input_dim = X.shape[1]  # 768 (BERT) + 12 (numerical)
model = BertRegressionModel(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Prepare DataLoader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Step 4: Train Model
model.train()
for epoch in range(5):  # Limited epochs for synthetic data
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
y_pred = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        y_pred.append(outputs.cpu().numpy())
y_pred = np.vstack(y_pred).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Performance: Mean Squared Error (MSE) = {mse:.2f}")

# Step 5: Predict CWV Scores
model.eval()
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    predicted_scores = model(X_tensor).cpu().numpy().flatten()
df['predicted_cwv_score'] = predicted_scores

# Step 6: Feature Importance (for numerical features only)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(num_data, y)
feature_importance = pd.DataFrame({
    'feature': num_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance (Numerical Features):")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance for Numerical Features (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'.")

# Step 7: Website Rankings
# Top 12 websites with highest CWV scores
top_12_high_cwv = df[df['predicted_cwv_score'] > 80][
    ['website_id', 'description', 'lcp', 'fid', 'cls', 'predicted_cwv_score']
].sort_values('predicted_cwv_score', ascending=False).head(12)
print("\nTop 12 Websites with High CWV Scores (Likely to Perform Well in SERP):")
print(top_12_high_cwv.round(4))

# Top 12 websites with lowest CWV scores
top_12_low_cwv = df[df['predicted_cwv_score'] < 50][
    ['website_id', 'description', 'lcp', 'fid', 'cls', 'predicted_cwv_score']
].sort_values('predicted_cwv_score').head(12)
print("\nTop 12 Websites with Low CWV Scores (Poor SERP Performance):")
print(top_12_low_cwv.round(4))

# Save results
top_12_high_cwv.to_csv('top_12_high_cwv_websites.csv', index=False)
top_12_low_cwv.to_csv('top_12_low_cwv_websites.csv', index=False)
df.to_csv('hotel_cwv_dataset.csv', index=False)
print("\nResults saved to 'top_12_high_cwv_websites.csv', 'top_12_low_cwv_websites.csv', and 'hotel_cwv_dataset.csv'.")