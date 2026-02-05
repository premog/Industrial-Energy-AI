import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

from dataset import create_data
from predict import predict_energy
from utils import calculate_cost

st.set_page_config(layout="wide")

st.title("⚡ Industrial Energy AI")

df = create_data()

# 🔥 Create anomaly column automatically
X = df.drop("Energy_Consumption",axis=1)
iso = IsolationForest(contamination=0.05,random_state=42)
df["Anomaly"] = iso.fit_predict(X)

# Metrics
col1,col2,col3 = st.columns(3)
col1.metric("Avg Energy",round(df["Energy_Consumption"].mean(),2))
col2.metric("Max Energy",round(df["Energy_Consumption"].max(),2))
col3.metric("Rows",len(df))

st.markdown("---")

# Prediction Panel
c1,c2 = st.columns(2)

with c1:
    prod = st.slider("Production Volume",0.5,1.0,0.75)
    hours = st.slider("Machine Hours",500,900,700)
    temp = st.slider("Temperature",200,450,300)

with c2:
    workers = st.slider("Workers",100,250,150)
    maint = st.slider("Maintenance Score",3,7,5)
    pf = st.slider("Power Factor",0.0,0.4,0.25)

input_df = pd.DataFrame([[prod,hours,temp,workers,maint,pf]],
columns=df.drop(["Energy_Consumption","Anomaly"],axis=1).columns)

if st.button("Predict Energy"):
    prediction = predict_energy(input_df)
    cost = calculate_cost(prediction)

    st.success(f"Energy: {prediction:.2f}")
    st.info(f"Cost ₹: {cost:.2f}")

# Scatter Plot
fig = plt.figure()
plt.scatter(df["Production_Volume"],df["Energy_Consumption"])
st.pyplot(fig)

# Anomaly Panel
st.subheader("🚨 Inefficient Operations")
anomalies = df[df["Anomaly"]==-1]
st.write(len(anomalies),"cases found")
st.dataframe(anomalies.head())

# Heatmap
st.subheader("🔥 Correlation Heatmap")
fig2 = plt.figure()
sns.heatmap(df.corr(),annot=False)
st.pyplot(fig2)