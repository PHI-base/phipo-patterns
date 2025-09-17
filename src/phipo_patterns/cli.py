# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import argparse
import tempfile

from phipo_patterns import patterns


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


def run(args):
    args = parse_args(args)
    with tempfile.TemporaryDirectory() as temp_dir:
        patterns.update_phipo_patterns(
            phipo_dir=args.ontology_dir,
            upheno_dir=args.upheno_dir,
            mapping_path=args.mapping_file,
            robot_path=args.robot_path,
            id_label_mapping_dir=temp_dir,
        )
