# train_model.py
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# load example dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# (optional) simple preprocessing example: create one-hot column example (none here)
# Save model columns for reindexing later
model_columns = list(X.columns)

# train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# save model and columns
joblib.dump(model, 'saved_model.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

print("Saved model to saved_model.pkl and model_columns.pkl")
