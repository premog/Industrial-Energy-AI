from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_ai(df):

    X = df.drop("Energy_Consumption",axis=1)
    y = df["Energy_Consumption"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train,y_train)

    score = r2_score(y_test,model.predict(X_test))

    # 🔥 REAL anomaly detection
    iso = IsolationForest(contamination=0.05,random_state=42)
    df["Anomaly"] = iso.fit_predict(X)

    return model,score,df