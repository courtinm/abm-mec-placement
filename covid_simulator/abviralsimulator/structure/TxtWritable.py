from abc import ABC, abstractmethod


class TxtWritable(ABC):

    @abstractmethod
    def get_text(self):
        pass
