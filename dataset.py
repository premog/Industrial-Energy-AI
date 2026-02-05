import pandas as pd

def create_data():

    df = pd.read_csv("energy_efficiency_data.csv")

    df = df.rename(columns={
        "Relative_Compactness":"Production_Volume",
        "Surface_Area":"Machine_Hours",
        "Wall_Area":"Temperature",
        "Roof_Area":"Workers",
        "Overall_Height":"Maintenance_Score",
        "Glazing_Area":"Power_Factor",
        "Heating_Load":"Energy_Consumption"
    })

    df = df[[
        "Production_Volume",
        "Machine_Hours",
        "Temperature",
        "Workers",
        "Maintenance_Score",
        "Power_Factor",
        "Energy_Consumption"
    ]].copy()

    df.dropna(inplace=True)

    return df