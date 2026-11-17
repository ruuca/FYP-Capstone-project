import os
import pickle
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- IMPORTANT: You need a small dataset to train on ---
# Since GTZAN is 1GB, let's use a tiny built-in workaround for demonstration.
# For your capstone, you can download 10 songs per era manually.
# But for NOW, we create a "dummy" classifier with realistic logic 
# so your code shows AI inference loading.

print("Creating a lightweight AI Era Classifier...")

# === REAL SOLUTION: Use your existing rule-based logic as a "mock" training ===
# Since we can't download 1000 songs right now, we will train a classifier
# on synthetic data that mimics your rule-based thresholds.
# This will let you load a real .pkl model today.

def generate_synthetic_data(n_samples=1000):
    X = []  # Features: [rolloff, high_energy, dynamic_range, flatness]
    y = []  # Labels: 0=1930s, 1=1960s, 2=1970s
    
    for _ in range(n_samples):
        # Simulate random audio features
        rolloff = np.random.uniform(2000, 18000)
        high_e = np.random.uniform(0.001, 0.2)
        dynamic = np.random.uniform(3, 30)
        flatness = np.random.uniform(0.001, 0.05)
        
        features = [rolloff, high_e, dynamic, flatness]
        
        # Label based on realistic rules (so it works with your audio)
        if rolloff < 6000 and high_e < 0.02:
            label = 0  # 1930s
        elif rolloff < 10000 and dynamic > 8:
            label = 1  # 1960s
        else:
            label = 2  # 1970s
            
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

X, y = generate_synthetic_data(2000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Test accuracy (should be high because rules are deterministic)
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the model
with open('era_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved to 'era_classifier.pkl'")
print("Place this file in your project root.")