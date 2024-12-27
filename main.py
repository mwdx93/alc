from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss  # Import the log_loss function
from sklearn.datasets import load_breast_cancer

from .alc import ALC 

# Load breast cancer dataset
dataset = load_breast_cancer()
X = dataset["data"]
y = dataset["target"]

encoder = OneHotEncoder(sparse=False)  # Set sparse to False for dense output
y_encoded = encoder.fit_transform(y.reshape(-1, 1))  # Fit and transform labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, shuffle=True)

# Instantiate ALC and fit on training data
alc = ALC(detoxification_cycles=500, detoxification_power=15)
liver = alc.fit(X_train, y_train)

# Predict on test data
pred = liver.reaction(X_test)

# Calculate log loss
loss = log_loss(y_test, pred) 
print("Log Loss:", loss)
