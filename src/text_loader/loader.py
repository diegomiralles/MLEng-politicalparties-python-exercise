import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import yaml

from typing import Optional

class DataLoader:
    def __init__(self, 
                 filepath:str="data/Tweets.csv",
                 vectorizer: Optional[TfidfVectorizer] = None,
                 encoder: Optional[LabelEncoder] = None):

        self.filepath = filepath
        self.data = self.load_data()
        self.vectorizer = vectorizer
        self.encoder = encoder
    @classmethod
    def from_config(cls, config_path: str):

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        vectorizer_config = config["text_loader"]["vectorizer"]
        vectorizer = TfidfVectorizer(**vectorizer_config)

        encoder = LabelEncoder()
        loader = cls(filepath=config["input_data_path"],
                     vectorizer=vectorizer,
                     encoder=encoder)
        return loader

    def load_data(self):
        """Loads data from a CSV file."""
        return pd.read_csv(self.filepath)

    @staticmethod
    def remove_characters(text: str) -> str:
        """Remove non-letters from a given string"""
        remove_chars = string.punctuation + string.digits + string.whitespace
        translator = str.maketrans('', '', remove_chars)
        return text.translate(translator)

    def clean_text(self, text: str) -> str:
        """Keep only retain words in a given string"""
        text = str(text) if text is not None else ""
        
        url_pattern = r'https?://\S+|www\.\S+'
        text = re.sub(url_pattern, '', text)    
        text = self.remove_characters(text)
        return text

    def vectorize_text(self, tweets: list[str]):
        return self.vectorizer.fit_transform(tweets).toarray() # type: ignore

    def label_encoder(self, parties: pd.Series):
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text) # type: ignore
        return self.vectorize_text(self.data.Tweet.values) # type: ignore

    def preprocess_parties(self):
        self.data.Party = self.data.Party.apply(self.clean_text) # type: ignore
        return self.label_encoder(self.data.Party.values) # type: ignore

