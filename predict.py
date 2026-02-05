import joblib

model = joblib.load("model.joblib")

def predict_energy(input_df):
    return model.predict(input_df)[0]