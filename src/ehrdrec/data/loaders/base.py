from abc import ABC
import polars as pl

class BaseLoader(ABC):
    def __init__(self, source):
        self.source = source
    
    def load(self) -> pl.DataFrame:
        raise NotImplementedError