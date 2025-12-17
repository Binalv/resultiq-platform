import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
df = pd.read_csv("data/resultiq_student_data.csv")

X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("ResultIQ model trained and saved successfully")


