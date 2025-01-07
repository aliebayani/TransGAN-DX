import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data['target'] = (data['num'] > 0).astype(int)
    data = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang',
                 'oldpeak', 'slope', 'ca', 'thal', 'target']]
    data.dropna(inplace=True)
    data = pd.get_dummies(data, columns=['sex', 'cp', 'restecg', 'slope', 'thal'])

    scaler = StandardScaler()
    data[['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']] = scaler.fit_transform(
        data[['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']]
    )

    X = data.drop(columns=['target'])
    y = data['target']

    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)
