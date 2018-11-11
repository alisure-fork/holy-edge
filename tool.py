import os
import yaml
import wget
from pyunpack import Archive
from termcolor import colored
from time import strftime, localtime


class Tool(object):

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        pass

    def read_yaml_file(self, config_file):
        try:
            p_file = open(config_file)
            d = yaml.load(p_file)
            p_file.close()
            return d
        except Exception as err:
            self.print_error('Error reading config file {}, {}'.format(config_file, err))
            pass
        pass

    def download_data(self, file_path, output_dir):
        _, rar_file = os.path.split(file_path)
        rar_file = os.path.join(output_dir, rar_file)

        if not os.path.exists(rar_file):
            self.print_info('Downloading {} to {}'.format(file_path, rar_file))
            _ = wget.download(file_path, out=output_dir)
            pass

        self.print_info('Decompressing {} to {}'.format(rar_file, output_dir))
        Archive(rar_file).extractall(output_dir)
        pass

    def print_info(self, info_string):
        info = '[{0}][INFO] {1}'.format(self.get_local_time(), info_string)
        print(colored(info, 'green'))

    def print_warning(self, warning_string):
        warning = '[{0}][WARNING] {1}'.format(self.get_local_time(), warning_string)
        print(colored(warning, 'blue'))

    def print_error(self, error_string):
        error = '[{0}][ERROR] {1}'.format(self.get_local_time(), error_string)
        print(colored(error, 'red'))

    @staticmethod
    def get_local_time():
        return strftime("%d %b %Y %Hh%Mm%Ss", localtime())

    @staticmethod
    def read_file_list(file_list):
        p_file = open(file_list)
        file_names = p_file.readlines()
        p_file.close()
        file_names = [f.strip() for f in file_names]
        return file_names

    @staticmethod
    def split_pair_names(file_names, base_dir):
        file_names = [c.split(' ') for c in file_names]
        return [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in file_names]

    pass
