import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def split_data(data, test_size=0.2):
    X = data['text_column']
    y = data['label_column']
    return train_test_split(X, y, test_size=test_size)
