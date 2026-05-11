# -----------------------------------------------------------------------------
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# You are free to: Share — copy and redistribute the material in any medium or format
#                  Adapt — remix, transform, and build upon the material for any purpose
#                  The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
# Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
# You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
#
# Created by Mahmood A. Jumaah and Tarik A. Rashid 2025
#
# Cite as:
#
# Jumaah MA, Ali YH, Rashid TA. Artificial liver classifier: A new alternative to conventional machine learning models.
# Frontiers in Artificial Intelligence. 2025 Aug 8;8:1639720. DOI: https://doi.org/10.48550/arXiv.2501.08074
#
# Jumaah MA, Ali YH, Rashid TA. 2024. Q-FOX Learning:
# QF-tuner: Breaking Tradition in Reinforcement Learning. DOI: https://doi.org/10.48550/arXiv.2402.16562
#
# Jumaah MA, Ali YH, Rashid TA, Vimal S. FOXANN: A Method for Boosting Neural Network Performance.
# Journal of Soft Computing and Computer Applications. 2024;1(1):2.DOI: https://doi.org/10.48550/arXiv.2407.03369
#
# Jumaah MA, Ali YH, Rashid TA (2025) An improved FOX optimization algorithm using adaptive exploration and exploitation
# for global optimization. PLOS ONE 20(9): e0331965. DOI: https://doi.org/10.1371/journal.pone.0331965

# Jumaah MA, Ali YH, Rashid TA. Efficient Q-learning hyperparameter tuning using FOX optimization algorithm.
# Results in Engineering. 2025 Mar 1;25:104341. DOI: https://doi.org/10.1016/j.rineng.2025.104341
# -----------------------------------------------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss  # Import the log_loss function
from sklearn.datasets import load_breast_cancer

from alc import ALC

# Load breast cancer dataset
dataset = load_breast_cancer()
X = dataset["data"]
y = dataset["target"]

encoder = OneHotEncoder(sparse_output=False)  # Set sparse to False for dense output
y_encoded = encoder.fit_transform(y.reshape(-1, 1))  # Fit and transform labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, shuffle=True
)

# Instantiate ALC and fit on training data
alc = ALC(opt_iterations=500, pop_size=15, lobules=10)
liver = alc.fit(X_train, y_train)

# Predict on test data
pred = liver.reaction(X_test)

# Calculate log loss
loss = log_loss(y_test, pred)
print("Log Loss:", loss)
