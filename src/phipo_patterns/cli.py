# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import argparse
import tempfile

from phipo_patterns import patterns, synchronize


def parse_args(args: list[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param args: command-line arguments from ``sys.argv``.
    :type args: list[str]
    :returns: an ``argparse.Namespace`` containing parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog='phipo_patterns',
        description='Manage the PHIPO pattern mapping table and phenotype patterns.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    parser_patterns = subparsers.add_parser(
        name='make_patterns',
        description=(
            'Make pattern data files from the PHIPO pattern mapping table and update the list of external patterns.'
        ),
    )
    parser_sync = subparsers.add_parser(
        name='sync_mapping',
        description='Synchronize the PHIPO pattern mapping table with the latest version of PHIPO and the latest patterns from uPheno.',
    )

    # arguments for make_patterns
    parser_patterns.add_argument(
        '--mapping-file',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the PHIPO pattern mapping CSV file',
    )
    parser_patterns.add_argument(
        '--ontology-dir',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the PHIPO repository',
    )
    parser_patterns.add_argument(
        '--upheno-dir',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the uPheno repository',
    )
    parser_patterns.add_argument(
        '--robot-path',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the ROBOT JAR file',
    )

    # arguments for sync_spreadsheet
    parser_sync.add_argument(
        '--mapping-file',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the PHIPO pattern mapping CSV file',
    )
    parser_sync.add_argument(
        '--ontology-dir',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the PHIPO repository',
    )
    parser_sync.add_argument(
        '--upheno-dir',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the uPheno repository',
    )
    parser_sync.add_argument(
        '--output',
        metavar='PATH',
        type=str,
        required=True,
        help='path to write the updated PHIPO pattern mapping CSV file',
    )

    return parser.parse_args(args)


def run(args: list[str]) -> None:
    """
    Run the command-line for the phipo_patterns package.

    :param args: command-line arguments from ``sys.argv``.
    :type args: list[str]
    """
    parsed_args = parse_args(args)
    match parsed_args.command:
        case 'make_patterns':
            with tempfile.TemporaryDirectory() as temp_dir:
                patterns.update_phipo_patterns(
                    phipo_dir=parsed_args.ontology_dir,
                    upheno_dir=parsed_args.upheno_dir,
                    mapping_path=parsed_args.mapping_file,
                    robot_path=parsed_args.robot_path,
                    id_label_mapping_dir=temp_dir,
                )
        case 'sync_spreadsheet':
            synchronize.sync_term_mapping_table(
                spreadsheet_path=parsed_args.mapping_file,
                phipo_dir=parsed_args.ontology_dir,
                upheno_dir=parsed_args.upheno_dir,
                out_path=parsed_args.output,
            )
        case _:
            # argparse should prevent this from being reached
            raise ValueError(f'unknown command: {parsed_args.command}')
