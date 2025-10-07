from abc import ABC, abstractmethod


class CsvWritable(ABC):

    @abstractmethod
    def get_header(self):
        pass

    @abstractmethod
    def get_row(self, index):
        pass

    @abstractmethod
    def get_row_size(self):
        pass
