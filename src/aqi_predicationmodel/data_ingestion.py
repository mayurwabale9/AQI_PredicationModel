import pandas as pd

def load_data():
    df = pd.read_excel(r'C:\AQI_PredicationModel\data\AirQualityUCI.xlsx')
    
    return df