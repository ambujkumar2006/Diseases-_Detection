import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('heart.csv')

# Encode categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoders = {}  # store label encoders for use during user input

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Prepare features and labels
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, validation_split=0.1)

# Evaluate
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Accept user input for prediction
print("\nYou can now enter new patient data for prediction.")

# Get column names
input_columns = list(X.columns)

while True:
    print("\n--- Enter patient data (or type 'exit' to quit) ---")
    user_input = []

    for col in input_columns:
        val = input(f"{col}: ")
        if val.lower() == 'exit':
            exit()

        # Encode categorical input
        if col in encoders:
            try:
                val_encoded = encoders[col].transform([val])[0]
            except:
                print(f"Invalid value for {col}. Allowed: {list(encoders[col].classes_)}")
                break
            user_input.append(val_encoded)
        else:
            user_input.append(float(val))

    if len(user_input) != len(input_columns):
        print("Invalid input. Try again.")
        continue

    # Scale input and predict
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)[0][0]
    print("Prediction:", "❌ Heart Disease Detected" if prediction > 0.5 else "✅ No Heart Disease")
