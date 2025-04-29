import requests

# Create a fake test input
features = {
    "GDP_per_Capita": 1.2,
    "Social_Support": 1.1,
    "Healthy_Life_Expectancy": 65.0,
    "Freedom": 0.8,
    "Generosity": 0.2,
    "Corruption_Perception": 0.3,
    "Unemployment_Rate": 5.0,
    "Education_Index": 0.85,
    "Population": 5000000,
    "Urbanization_Rate": 70.0,
    "Life_Satisfaction": 7.0,
    "Public_Trust": 0.6,
    "Mental_Health_Index": 75.0,
    "Income_Inequality": 0.25,
    "Public_Health_Expenditure": 6.5,
    "Climate_Index": 68.0,
    "Work_Life_Balance": 7.5,
    "Internet_Access": 85.0,
    "Crime_Rate": 20.0,
    "Political_Stability": 0.7,
    "Employment_Rate": 95.0
}

url = 'http://127.0.0.1:8000/predict'

response = requests.post(url, json=features)
print(response.json())