from dataset import create_data
from sklearn.ensemble import RandomForestRegressor
import joblib

df = create_data()

X = df.drop("Energy_Consumption",axis=1)
y = df["Energy_Consumption"]

model = RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(X,y)

joblib.dump(model,"model.joblib")

print("Model trained using REAL dataset.")