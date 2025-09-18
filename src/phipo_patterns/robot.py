# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import os
import shlex
import subprocess
from typing import Optional


class Robot:
    """
    A class to interact with ROBOT, a command-line tool for working with Open Biomedical Ontologies.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """
        Initialize the Robot instance with the path to the robot.jar file.

        :param path: path to the robot.jar file. If ``None``, attempts to locate robot.jar in the system PATH.
        :type path: Optional[str]
        """
        if path is None:
            self.__path = self.__get_robot_path()
        else:
            self.__path = path

    def __get_robot_path(self) -> str:
        """
        Search for the robot.jar file in the system PATH.

        :returns: full path to the robot.jar file.
        :rtype: str
        :raises FileNotFoundError: if robot.jar is not found in any PATH directory ending with 'robot'.
        """
        env_path = os.environ['PATH']
        robot_paths = (path for path in env_path.split(';') if path.endswith('robot'))
        for directory in robot_paths:
            if 'robot.jar' in os.listdir(directory):
                return os.path.join(directory, 'robot.jar')
        raise FileNotFoundError('Cannot find robot.jar on PATH')

    def __run(self, arg_string: str) -> None:
        """
        Run a ROBOT command using the ``subprocess`` module.

        :param arg_string: the command-line arguments to pass to ROBOT.
        :type arg_string: str
        """
        args = shlex.split(arg_string, posix=False)
        subprocess.run(['java', '-jar', self.__path, *args], check=True)

    def convert(self, input_path: str, output_path: str) -> None:
        """
        Convert an ontology file to another format.

        :param input_path: path to the input ontology file.
        :type input_path: str
        :param output_path: path to the converted output file. The output format will be inferred from the output file extension.
        :type output_path: str
        """
        self.__run(f'convert --input {input_path} --output {output_path}')

    def query(self, query: str, input_path: str, output_path: str) -> None:
        """
        Execute a SPARQL query on an ontology file.

        :param query: path to the SPARQL query file.
        :type query: str
        :param input_path: path to the input ontology file.
        :type input_path: str
        :param output_path: path to the output file. The output format will be inferred from the output file extension.
        :type output_path: str
        """
        self.__run(f'query --input {input_path} --query {query} {output_path}')

    def export(
        self, input_path: str, output_path: str, header: str = 'ID|LABEL'
    ) -> None:
        """
        Export classes from an ontology file to a tabular format.

        :param input_path: path to the input ontology file.
        :type input_path: str
        :param output_path: path to the output file. The output format will be inferred from the output file extension.
        :type output_path: str
        :param header: pipe-separated list of special keywords or properties to export as columns. Defaults to 'ID|LABEL'.
        :type header: str
        """
        args = (
            'export',
            f'--input {input_path}',
            f'--header {header}',
            '--include "classes"',
            f'--export {output_path}',
        )
        command = ' '.join(args)
        self.__run(command)
