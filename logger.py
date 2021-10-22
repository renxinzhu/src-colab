from typing import Union, Dict
from datetime import datetime
import os

#should be moved to hyper_parameters.py later
LOG_PATH = "./log"

class Logger:
    def __init__(self, client_id: Union[int, None]):
        self.client_id = client_id
        self.log_path = LOG_PATH
        self.name = f'client-{self.client_id}' if self.client_id else 'server'
        filename = f'{self.name}-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.txt'
        self.filepath = os.path.join(self.log_path, filename)

    def print(self, message):
        print(f'[{datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}]' +
              f'[{self.name}] ' +
              f'{message}')

    def save_current_state(self, data):

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        with open(self.filepath, 'a+') as outfile:
            content = [str(i) for i in data.values()]
            outfile.write(",".join(content))
            outfile.write('\n')
