import pandas as pd




class PandasReader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self):
        """Loads data from a CSV file."""
        return pd.read_csv(self.filepath)
        
    def apply(self, func: callable):
        """Applies a function to the data."""
        self.data = self.data.apply(func, axis=1)