from typing import Optional,Protocol


class DataLoaderProtocol(Protocol):
    def preprocess_tweets(self, clean_text: bool = False):
        ...

    def preprocess_parties(self, clean_text: bool = False):
        ...
class VectorizerProtocol(Protocol):
    def fit_transform(self, tweets: list[str]):
        ...
class DataReaderProtocol(Protocol):
    def load_data(self):
        ...
    def apply(self, func: callable):
        ...
