from abc import ABC, abstractmethod

import polars as pl


class BaseLoader(ABC):
    def __init__(self, source: str) -> None:
        self.source = source

    @abstractmethod
    def load(self) -> pl.DataFrame:
        raise NotImplementedError
