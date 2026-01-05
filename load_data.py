import pandas as pd

def load_data(path):
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.str.contains('Unnamed', na=False)]
    
    # Ajouter la colonne URL
    data['url'] = 'https://www.flickr.com/photos/' + data['user'].astype(str) + '/' + data['id'].astype(str)
    
    return data
