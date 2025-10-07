import csv
from os import path
from pathlib import Path
from abviralsimulator.structure import CsvWritable, TxtWritable


class Writer:

    def __init__(self, data_store):
        self.data_store = data_store

    def __create_folder(self, folder_name):
        Path(folder_name).mkdir(parents=True, exist_ok=True)

    def write_to_csv(self, folder_name):
        modules = self.data_store.get_modules()
        for name, module in modules.items():

            if not isinstance(module, CsvWritable):
                continue

            self.__create_folder(folder_name)
            full_path = path.join(folder_name, name)
            full_path += ".csv"
            with open(full_path, "w") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(module.get_header())
                for i in range(module.get_row_size()):
                    csv_writer.writerow(module.get_row(i))

    def write_to_txt(self, folder_name):
        modules = self.data_store.get_modules()
        for name, module in modules.items():

            if not isinstance(module, TxtWritable):
                continue

            self.__create_folder(folder_name)
            full_path = path.join(folder_name, name)
            full_path += ".txt"
            with open(full_path, "w") as file:
                file.write(module.get_text())



