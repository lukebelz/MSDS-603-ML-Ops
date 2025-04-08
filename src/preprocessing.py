import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split

# Column names
col_names = [
    "Country",
    "Year",
    "Happiness_Score",
    "GDP_per_Capita",
    "Social_Support",
    "Healthy_Life_Expectancy",
    "Freedom",
    "Generosity",
    "Corruption_Perception",
    "Unemployment_Rate",
    "Education_Index",
    "Population",
    "Urbanization_Rate",
    "Life_Satisfaction",
    "Public_Trust",
    "Mental_Health_Index",
    "Income_Inequality",
    "Public_Health_Expenditure",
    "Climate_Index",
    "Work_Life_Balance",
    "Internet_Access",
    "Crime_Rate",
    "Political_Stability",
    "Employment_Rate"
]

# Load and split
data = pd.read_csv('data/world_happiness_report.csv', names=col_names)

# Drop rows with entirely missing values
data = data.dropna(how='all')

# Target can be anything – here we'll predict Happiness_Score
target = "Happiness_Score"
X = data.drop(columns=[target])
y = data[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selectors
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Full preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Combine preprocessor into full pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

# Fit + transform
X_train_transformed = clf.fit_transform(X_train)
X_test_transformed = clf.transform(X_test)

# Get feature names after transformation (optional, for DataFrame reconstruction)
ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
new_feature_names = numeric_features + list(cat_feature_names)

# Save as DataFrames
train_new = pd.DataFrame(X_train_transformed, columns=new_feature_names)
test_new = pd.DataFrame(X_test_transformed, columns=new_feature_names)

# Optionally, reattach target for supervised learning
train_new[target] = y_train.values
test_new[target] = y_test.values

# Save CSVs
train_new.to_csv('data/processed_world_happiness_train_data.csv', index=False)
test_new.to_csv('data/processed_world_happiness_test_data.csv', index=False)

# Save pipeline
with open('data/pipeline.pkl','wb') as f:
    pickle.dump(clf, f)