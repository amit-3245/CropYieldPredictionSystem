import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("dataset/crop_production.csv")

# Remove missing values
data = data.dropna()

# Remove zero area rows
data = data[data['Area'] > 0]

# Remove unnecessary columns
data = data.drop(['District_Name','Season','Crop_Year'], axis=1)

# Create Yield column (Target Variable)
data['Yield'] = data['Production'] / data['Area']

#  Add dummy weather columns 
data['Temperature'] = 25
data['Humidity'] = 60
data['Rainfall'] = 100

# Encode categorical data
le_state = LabelEncoder()
le_crop = LabelEncoder()

data['State_Name'] = le_state.fit_transform(data['State_Name'])
data['Crop'] = le_crop.fit_transform(data['Crop'])

# Features and Target
X = data[['State_Name','Crop','Area','Temperature','Humidity','Rainfall']]
y = data['Yield']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Lightweight Random Forest Model (RAM Optimized)
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Save Model with Compression
joblib.dump(model, 'model/crop_yield_model.pkl', compress=3)
joblib.dump(le_state, 'model/state_encoder.pkl', compress=3)
joblib.dump(le_crop, 'model/crop_encoder.pkl', compress=3)

print("Weather-Based Model Saved Successfully!")