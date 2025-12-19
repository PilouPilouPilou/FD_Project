import pandas as pd

def load_data(path):
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.str.contains('Unnamed', na=False)]
    return data
