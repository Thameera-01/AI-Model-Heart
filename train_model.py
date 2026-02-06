"""
Training script for heart disease prediction model.
This script creates a sample model and scalers for demonstration purposes.
In production, this should be trained with actual data.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate sample data for demonstration
# In production, load from actual CSV file: pd.read_csv('cardio_dataset.csv')
np.random.seed(42)
n_samples = 1000

# Generate synthetic data with 7 features: gender, age, TC, HDL, smoke, bpm, diabetes
# Features: gender (0/1), age (20-80), TC (100-300), HDL (20-100), smoke (0/1), bpm (0/1), diab (0/1)
data = np.column_stack([
    np.random.randint(0, 2, n_samples),  # gender
    np.random.randint(20, 81, n_samples),  # age
    np.random.uniform(100, 300, n_samples),  # TC
    np.random.uniform(20, 100, n_samples),  # HDL
    np.random.randint(0, 2, n_samples),  # smoke
    np.random.randint(0, 2, n_samples),  # bpm
    np.random.randint(0, 2, n_samples),  # diabetes
])

# Generate target (risk score 0-1)
# Higher age, TC, smoking, high BP, diabetes increase risk
target = (
    (data[:, 1] / 100) * 0.3 +  # age factor
    (data[:, 2] / 400) * 0.2 +  # TC factor
    (1 - data[:, 3] / 100) * 0.2 +  # HDL factor (inverse)
    data[:, 4] * 0.1 +  # smoking
    data[:, 5] * 0.1 +  # high BP
    data[:, 6] * 0.1 +  # diabetes
    np.random.normal(0, 0.05, n_samples)  # noise
)
target = np.clip(target, 0, 1).reshape(-1, 1)

print("Data shape:", data.shape)
print("Target shape:", target.shape)

# Scale the data
scaler_data = MinMaxScaler()
scaler_target = MinMaxScaler()

scaler_data.fit(data)
scaler_target.fit(target)

data_scaled = scaler_data.transform(data)
target_scaled = scaler_target.transform(target)

# Split data
train_data, test_data, train_target, test_target = train_test_split(
    data_scaled, target_scaled, test_size=0.2, random_state=42
)

print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# Build model
model = Sequential()
model.add(Dense(200, input_dim=7, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.summary()

# Train model
print("\nTraining model...")
history = model.fit(
    train_data, train_target,
    validation_data=(test_data, test_target),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate
from sklearn.metrics import r2_score
y_pred = model.predict(test_data)
r2 = r2_score(test_target, y_pred)
print(f'\nR2 Score: {r2:.4f}')

# Save model and scalers
print("\nSaving model and scalers...")
model.save('models/heart_model.h5')
joblib.dump(scaler_data, 'models/scaler_data.sav')
joblib.dump(scaler_target, 'models/scaler_target.sav')

print("Model and scalers saved successfully!")
print("Files created:")
print("  - models/heart_model.h5")
print("  - models/scaler_data.sav")
print("  - models/scaler_target.sav")
