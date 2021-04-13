import os

class Config:
    root = os.getcwd()
    output_directory = f'{root}\output'
    input_directory = f'{root}\input'

    def __init__(self):
        pass

    def get_config(self):
        return {
            "root": self.root,
            "output_directory": self.output_directory,
            "input_directory": self.input_directory
        }
        