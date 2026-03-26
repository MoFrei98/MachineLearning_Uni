from abc import ABC, abstractmethod

class Kernel(ABC):
    @abstractmethod
    def get_params(self) -> dict:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass