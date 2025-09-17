# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import os
import shlex
import subprocess


class Robot:

    def __init__(self, path=None):
        if path == None:
            self.__path = self.__get_robot_path()
        else:
            self.__path = path

    def __get_robot_path(self):
        env_path = os.environ['PATH']
        robot_paths = (path for path in env_path.split(';') if path.endswith('robot'))
        for directory in robot_paths:
            if 'robot.jar' in os.listdir(directory):
                return os.path.join(directory, 'robot.jar')
        else:
            raise FileNotFoundError('Cannot find robot.jar on PATH')

    def __run(self, arg_string):
        args = shlex.split(arg_string, posix=False)
        subprocess.run(['java', '-jar', self.__path, *args], check=True)

    def convert(self, input_path, output_path):
        self.__run(f'convert --input {input_path} --output {output_path}')

    def query(self, query, input_path, output_path):
        self.__run(f'query --input {input_path} --query {query} {output_path}')

    def export(self, input_path, output_path, header='ID|LABEL'):
        args = (
            'export',
            f'--input {input_path}',
            f'--header {header}',
            '--include "classes"',
            f'--export {output_path}',
        )
        command = ' '.join(args)
        self.__run(command)
