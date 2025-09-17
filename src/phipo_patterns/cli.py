# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import argparse


def parse_args(args: list[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param args: command-line arguments from ``sys.argv``.
    :type args: list[str]
    :returns: an ``argparse.Namespace`` containing parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog='uPheno pattern maker',
        description='Script to make phenotype patterns from the PHIPO pattern mapping spreadsheet.',
    )
    parser.add_argument(
        '--mapping-file',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the PHIPO pattern mapping CSV file',
    )
    parser.add_argument(
        '--ontology-dir',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the PHIPO repository',
    )
    parser.add_argument(
        '--upheno-dir',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the uPheno repository',
    )
    parser.add_argument(
        '--robot-path',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the ROBOT JAR file',
    )
    return parser.parse_args(args)
