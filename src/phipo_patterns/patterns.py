# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator, Mapping, Sequence

import pandas as pd
import yaml

pd.set_option('future.no_silent_downcasting', True)

OBO_ID_PATTERN = re.compile(r'^(?P<ns>[A-Za-z]+):(?P<id>\d+)$', re.ASCII)


def all_variables_imported(
    variables: pd.Series, import_namespaces: Collection[str]
) -> bool:
    """
    Test whether all ontology terms used in pattern variables are from an ontology
    imported in PHIPO.

    :param variables: a pandas Series of the variables column in the pattern mapping table.
    :type variables: pandas.Series
    :param import_namespaces: a collection of namespace prefixes for the ontologies imported by PHIPO.
    :type import_namespaces: Collection[str]
    :returns: ``True`` if all variable terms are from imported ontologies, otherwise ``False``.
    :rtype: bool
    """
    options = '|'.join(import_namespaces)
    pattern = re.compile(fr'(?P<ns>{options}):\d{{7}}')
    return variables.str.extractall(pattern)['ns'].isin(import_namespaces).all()


def load_pattern_mapping_file(path: str | Path) -> pd.DataFrame:
    """
    Load the PHIPO pattern mapping CSV file.

    :param path: path to the PHIPO pattern mapping CSV file.
    :type path: str | pathlib.Path
    :returns: the loaded mapping table.
    :rtype: pandas.DataFrame
    """
    return pd.read_csv(path)


def iter_pattern_files(dir_path: Path) -> Iterator[Path]:
    """
    Iterate over pattern definition YAML files in a directory.

    :param dir_path: directory to search for ``.yaml`` files.
    :type dir_path: pathlib.Path
    :yields: paths to ``.yaml`` files.
    :rtype: Iterator[pathlib.Path]
    """
    for child in dir_path.iterdir():
        if child.is_file() and child.suffix == '.yaml':
            yield child


def camel_case_to_lowercase(string: str) -> str:
    """
    Convert a camel case string to a space-separated lowercase string.

    :param string: a camel case string.
    :type string: str
    :returns: a space-separated lowercase string.
    :rtype: str
    """
    return re.sub('(?=[A-Z])', ' ', string).lower()


def lowercase_to_camel_case(string: str) -> str:
    """
    Convert a space-separated lowercase string to camel case.

    :param string: a space-separated lowercase string.
    :type string: str
    :returns: a camel case string.
    :rtype: str
    """
    words = string.split(' ')
    return ''.join((words[0].lower(), *(word.capitalize() for word in words[1:])))


def iter_upheno_pattern_definition_paths(upheno_dir_path: str | Path) -> Iterator[Path]:
    """
    Iterate over uPheno pattern definition files in the uPheno repository folder.

    Searches in ``/src/patterns`` in the following directories:
    ``dosdp-dev``, ``dosdp-patterns``, and ``dosdp-workshop``.

    :param upheno_dir_path: path to the local uPheno repository folder.
    :type upheno_dir_path: str | pathlib.Path
    :yields: paths to uPheno pattern definition files.
    :rtype: Iterator[pathlib.Path]
    """
    upheno_dir = Path(upheno_dir_path)
    pattern_root_dir = upheno_dir / 'src' / 'patterns'
    pattern_dirs = ('dosdp-dev', 'dosdp-patterns', 'dosdp-workshop')
    pattern_dir_paths = (pattern_root_dir / dir_path for dir_path in pattern_dirs)
    for dir_path in pattern_dir_paths:
        for file_path in iter_pattern_files(dir_path):
            yield file_path


def iter_normalized_pattern_names(pattern_paths: Iterable[Path]) -> Iterator[str]:
    """
    Yield normalized (space-separated lowercase) pattern names from file paths.

    :param pattern_paths: iterable of pattern file paths.
    :type pattern_paths: Iterable[pathlib.Path]
    :yields: normalized pattern names derived from file stems.
    :rtype: Iterator[str]
    """
    for path in pattern_paths:
        yield camel_case_to_lowercase(path.stem)


def assert_missing_patterns_have_errors(
    pattern_df: pd.DataFrame, pattern_names: Collection[str]
) -> None:
    """
    Assert that rows with undefined or invalid pattern names have an error code in the table.

    :param pattern_df: the PHIPO pattern mapping table.
    :type pattern_df: pandas.DataFrame
    :param pattern_names: a collection of valid normalized pattern names.
    :type pattern_names: Collection[str]
    :raises AssertionError: if any row contains an undefined or invalid pattern.
    """
    patterns = pattern_df.pattern
    error_codes = pattern_df.error
    invalid_pattern_index = (
        patterns.isna()
        | (patterns == '?')
        | (patterns == '-')
        | ~patterns.isin(pattern_names)
    )
    is_missing_no_pattern_error = ~error_codes.str.contains('NO_PATTERN', na=False)
    not_merged_or_unused = ~error_codes.isin(('MERGED', 'UNUSED'))
    rows_needing_error = error_codes[
        invalid_pattern_index & not_merged_or_unused & is_missing_no_pattern_error
    ]
    assert rows_needing_error.empty, rows_needing_error


def assert_missing_variables_have_errors(pattern_df: pd.DataFrame) -> None:
    """
    Assert that rows with missing or invalid variables have an error code in the table.

    :param pattern_df: the PHIPO pattern mapping table.
    :type pattern_df: pandas.DataFrame
    :raises AssertionError: if any row has a missing or invalid variable.
    """

    def is_variable_imported(variables, import_namespaces):
        options = '|'.join(import_namespaces)
        pattern = re.compile(fr'(?P<ns>{options}):\d{{7}}')
        has_imported_variable = (
            variables.str.extractall(pattern)['ns']
            .isin(import_namespaces)
            .groupby(level=0)
            .agg('all')  # Any false match makes the row False
            .reindex_like(variables)
            .fillna(True)  # Rows with no variables are True
        )
        return has_imported_variable

    import_namespaces = {'IDO', 'GO', 'RO', 'CHEBI', 'PATO', 'SO', 'CL'}
    variables = pattern_df.variables
    error_codes = pattern_df.error
    is_variable_not_obo_id = variables.str.extract(OBO_ID_PATTERN).isna().any(axis=1)
    invalid_variables_index = (
        variables.isna()
        | (variables == '?')
        | (variables == '-')
        | is_variable_not_obo_id
        | ~is_variable_imported(variables, import_namespaces)
    )
    is_missing_no_variable_error = ~error_codes.str.contains('NO_VARIABLE', na=False)
    not_merged_or_unused = ~error_codes.isin(('MERGED', 'UNUSED'))
    rows_needing_error = error_codes[
        invalid_variables_index & not_merged_or_unused & is_missing_no_variable_error
    ]
    assert rows_needing_error.empty, rows_needing_error


def make_variable_columns(
    variable_series: pd.Series,
    pattern_variables: Sequence[str],
    id_label_mapping: Mapping[str, str],
) -> pd.DataFrame:
    """
    Expand the variable column into separate ID and label columns.

    :param variable_series: the 'variables' column from the PHIPO pattern mapping table.
    :type variable_series: pandas.Series
    :param pattern_variables: names of variables expected by the pattern.
    :type pattern_variables: Sequence[str]
    :param id_label_mapping: mapping from term OBO ID to term label.
    :type id_label_mapping: Mapping[str, str]
    :returns: a DataFrame with ID and label columns for all variables.
    :rtype: pandas.DataFrame
    """
    split_df = variable_series.str.split('; ', expand=True)
    data: dict[str, pd.Series] = {}
    for variable, column in zip(pattern_variables, split_df.columns):
        id_column_name = variable
        label_column_name = variable + '_label'
        id_column = split_df[column]
        data[id_column_name] = id_column
        data[label_column_name] = id_column.map(id_label_mapping)
    return pd.DataFrame(data, index=variable_series.index)


def make_mapping_data_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make an intermediate mapping table from pattern filename to the columns required for the pattern data files.

    :param filtered_df: filtered PHIPO pattern mapping table.
    :type filtered_df: pandas.DataFrame
    :returns: the intermediate mapping table.
    :rtype: pandas.DataFrame
    """
    columns = ['pattern', 'term', 'label', 'variables']
    df = filtered_df[columns].copy()
    # Tried to use pd.Series.map here, but it didn't work
    df.pattern = df.pattern.apply(lowercase_to_camel_case)
    mapping_data_df = df.rename(
        columns={
            'pattern': 'path',
            'term': 'defined_class',
            'label': 'defined_class_label',
        }
    )
    return mapping_data_df


def make_data_file_dataframes(
    mapping_data_df: pd.DataFrame,
    patterns_info: Mapping[str, Mapping[str, Any]],
    id_label_mapping: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    """
    Make a mapping from a pattern data file name to a pattern data table, using data from the intermediate mapping table.

    :param mapping_data_df: the intermediate mapping table from :func:`make_mapping_data_df`.
    :type mapping_data_df: pandas.DataFrame
    :param patterns_info: the variables list and textual definition for each pattern, from :func:`get_patterns_info`.
    :type patterns_info: Mapping[str, Mapping[str, Any]]
    :param id_label_mapping: mapping from OBO ID to term label.
    :type id_label_mapping: Mapping[str, str]
    :returns: mapping from pattern data file name to pattern data table.
    :rtype: dict[str, pandas.DataFrame]
    """
    data_groups = mapping_data_df.groupby('path')
    filename_to_df_map: dict[str, pd.DataFrame] = {}
    for pattern_name, df in data_groups:
        pattern_name = str(pattern_name)
        pattern_info = patterns_info.get(pattern_name)
        if not pattern_info:
            print(f'pattern not found {pattern_name}')
            continue
        defined_class_df = df[['defined_class', 'defined_class_label']]
        variable_df = make_variable_columns(
            df.variables, pattern_info['variables'], id_label_mapping
        )
        data_df = pd.concat([defined_class_df, variable_df], axis=1).sort_values(
            'defined_class'
        )
        filename = pattern_name + '.tsv'
        filename_to_df_map[filename] = data_df
    return filename_to_df_map


def get_patterns_info(
    upheno_dir: str | Path, pattern_definition_file_names: Collection[str]
) -> dict[str, dict[str, Any]]:
    """
    Get variable lists and textual definitions from uPheno pattern definition files.

    :param upheno_dir: path to the local uPheno repository folder.
    :type upheno_dir: str | pathlib.Path
    :param pattern_definition_file_names: the filenames of pattern definition files to include.
    :type pattern_definition_file_names: Collection[str]
    :returns: mapping from pattern filename to definition and variables.
    :rtype: dict[str, dict[str, Any]]
    """
    pattern_definition_file_paths = iter_upheno_pattern_definition_paths(upheno_dir)
    patterns_info: dict[str, dict[str, Any]] = {}
    for pattern_definition_file_path in pattern_definition_file_paths:
        file_name = pattern_definition_file_path.name
        if file_name not in pattern_definition_file_names:
            continue
        with open(pattern_definition_file_path, encoding='utf-8') as yaml_file:
            pattern_definition = yaml.safe_load(yaml_file)
        # Use the name from the file path, since the name used within
        # the definition file is not consistent.
        name = pattern_definition_file_path.stem
        fmt_args = tuple(f'[{v}]' for v in pattern_definition['def']['vars'])
        text_definition = pattern_definition['def']['text'] % fmt_args
        info = {
            'definition': text_definition,
            'variables': list(pattern_definition['vars']),
        }
        patterns_info[name] = info
    return patterns_info


def filter_mapping_df(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the PHIPO pattern mapping table to exclude obsolete terms, rows with errors, and missing patterns.

    :param mapping_df: the PHIPO pattern mapping table.
    :type mapping_df: pandas.DataFrame
    :returns: the filtered PHIPO pattern mapping table.
    :rtype: pandas.DataFrame
    :raises AssertionError: if any obsolete term lacks an ``obsolete`` label prefix.
    """
    df = mapping_df.copy()
    # Ensure all obsolete terms are marked as obsolete
    has_obsolete_label = df.label.str.startswith('obsolete', na=False)
    assert df[has_obsolete_label].obsolete.all()
    df.obsolete = df.obsolete.fillna(False).astype(bool)
    filtered_df = df[df.error.isna() & df.pattern.notna()]
    return filtered_df


def write_ontology_term_labels(
    phipo_dir: Path | str, out_dir: Path | str, robot_path: Path | str
) -> None:
    """
    Export ontology term labels as CSV files from mirrored versions of the import ontologies in the PHIPO repository folder.

    Uses the ROBOT tool to export terms. One CSV file is created for each ontology.

    :param phipo_dir: path to the local PHIPO repository folder.
    :type phipo_dir: pathlib.Path | str
    :param out_dir: path to the output directory for the term label files.
    :type out_dir: pathlib.Path | str
    :param robot_path: path to the ROBOT JAR file.
    :type robot_path: pathlib.Path | str
    :raises subprocess.CalledProcessError: if the ``robot`` command fails.
    """

    def iter_ontology_mirror_paths(phipo_dir):
        mirror_dir = phipo_dir / 'src' / 'ontology' / 'mirror'
        for child in mirror_dir.iterdir():
            if child.suffix != '.owl':
                continue
            ontology_path = child
            yield ontology_path

    phipo_dir = Path(phipo_dir)
    out_dir = Path(out_dir)
    arg_dict = {
        'java': 'java',
        'opt_jar': '-jar',
        'jar_path': robot_path,
        'command:': 'export',
        'opt_input': '--input',
        'input': None,
        'opt_header': '--header',
        'header': 'ID|LABEL',
        'opt_export': '--export',
        'export': None,
    }
    for path in iter_ontology_mirror_paths(phipo_dir):
        arg_dict['input'] = str(path)
        arg_dict['export'] = (out_dir / path.stem).with_suffix('.csv')
        args = list(arg_dict.values())
        subprocess.run(args, check=True)


def load_id_label_mapping(term_file_paths: Iterable[Path]) -> dict[str, str]:
    """
    Build an ID to label mapping for the terms from the ontologies imported by PHIPO.

    :param term_file_paths: iterable of CSV files exported by ROBOT.
    :type term_file_paths: Iterable[pathlib.Path]
    :returns: mapping from OBO ID to label.
    :rtype: dict[str, str]
    """
    id_label_mapping: dict[str, str] = {}
    for path in term_file_paths:
        ontology_name = path.stem.upper()
        df = pd.read_csv(path)
        # Only get terms from the defining ontology, otherwise older
        # labels from other ontologies may override the label.
        is_own_term = df['ID'].str.startswith(ontology_name)
        is_label_notna = df['LABEL'].notna()
        indexer = is_own_term & is_label_notna
        mapping = df[indexer].set_index('ID')['LABEL'].to_dict()
        id_label_mapping.update(mapping)
    return id_label_mapping


def make_pattern_urls(
    pattern_filenames: Iterable[str], upheno_dir_path: str | Path
) -> list[str]:
    """
    Converts each pattern filename into a PURL for its uPheno pattern definition file.

    Currently only supports patterns from ``dosdp-patterns`` and ``dosdp-dev``.

    :param pattern_filenames: iterable of uPheno pattern filenames.
    :type pattern_filenames: Iterable[str]
    :param upheno_dir_path: path to the local uPheno repository folder.
    :type upheno_dir_path: str | pathlib.Path
    :returns: a sorted list of PURLs for the given patterns.
    :rtype: list[str]
    :raises KeyError: if a pattern name cannot be found in uPheno or lacks a PURL mapping.
    """
    upheno_pattern_paths = iter_upheno_pattern_definition_paths(upheno_dir_path)
    path_lookup = {path.stem: path for path in upheno_pattern_paths}
    url_fragment_lookup = {
        'dosdp-patterns': 'patterns',
        'dosdp-dev': 'patterns-dev',
    }
    pattern_urls: list[str] = []
    for filename in pattern_filenames:
        pattern_name = Path(filename).stem
        pattern_path = path_lookup.get(pattern_name)
        if pattern_path is None:
            raise KeyError(f'pattern name not found in uPheno patterns: {pattern_name}')
        pattern_dir = pattern_path.parent.stem
        url_fragment = url_fragment_lookup.get(pattern_dir)
        if url_fragment is None:
            raise KeyError(f'pattern has no PURL: {pattern_name}')
        url = (
            f'http://purl.obolibrary.org/obo/upheno/{url_fragment}/{pattern_name}.yaml'
        )
        pattern_urls.append(url)
    return sorted(pattern_urls)


def write_external_txt(path: Path | str, pattern_urls: Iterable[str]) -> None:
    """
    Write pattern definition file PURLs to the ``external.txt`` file in the PHIPO repository.

    :param path: path to the ``external.txt`` file.
    :type path: pathlib.Path | str
    :param pattern_urls: iterable of pattern definition file PURLs.
    :type pattern_urls: Iterable[str]
    """
    lines = (url.rstrip() + '\n' for url in pattern_urls)
    with open(path, 'w', encoding='utf-8') as external_txt:
        external_txt.writelines(lines)


def update_external_txt(
    external_txt_path: Path | str,
    pattern_filenames: Iterable[str],
    upheno_dir_path: Path | str,
) -> None:
    """
    Update the ``external.txt`` file in the PHIPO repository with the patterns used in the PHIPO pattern mapping table.

    :param external_txt_path: path to the ``external.txt`` file.
    :type external_txt_path: pathlib.Path | str
    :param pattern_filenames: iterable of pattern filenames.
    :type pattern_filenames: Iterable[str]
    :param upheno_dir_path: path to the local uPheno repository folder.
    :type upheno_dir_path: pathlib.Path | str
    """
    pattern_urls = make_pattern_urls(pattern_filenames, upheno_dir_path)
    write_external_txt(external_txt_path, pattern_urls)


def update_pattern_data_files(
    data_file_dataframes: Mapping[str, pd.DataFrame], pattern_data_dir: Path
) -> None:
    """
    Write pattern data files for each pattern to the PHIPO pattern data directory.

    :param data_file_dataframes: mapping of pattern data filenames to pattern data.
    :type data_file_dataframes: Mapping[str, pandas.DataFrame]
    :param pattern_data_dir: output directory for pattern data files.
    :type pattern_data_dir: pathlib.Path
    """
    for filename, df in data_file_dataframes.items():
        path = pattern_data_dir / filename
        df.to_csv(path, sep='\t', index=False)


def validate_data_file_labels(data_file_dataframes: Mapping[str, pd.DataFrame]) -> None:
    """
    Validate that all label columns in the pattern data files are fully populated.

    :param data_file_dataframes: mapping of pattern data filenames to pattern data.
    :type data_file_dataframes: Mapping[str, pandas.DataFrame]
    :raises AssertionError: if any label column contains missing values.
    """
    patterns_with_missing_labels: list[str] = []
    for key, df in data_file_dataframes.items():
        label_columns = df.columns[df.columns.str.endswith('_label')]
        if df[label_columns].isna().any().any():
            patterns_with_missing_labels.append(key)
    if patterns_with_missing_labels:
        lines = '\n'.join(patterns_with_missing_labels)
        assert False, f'patterns have missing labels:\n{lines}'


def get_pattern_file_names(
    filtered_df: pd.DataFrame, upheno_dir: Path | str
) -> set[str]:
    """
    Get the pattern filenames that are in both the PHIPO pattern mapping table and uPheno.

    :param filtered_df: filtered PHIPO pattern mapping table.
    :type filtered_df: pandas.DataFrame
    :param upheno_dir: path to the local uPheno repository folder.
    :type upheno_dir: pathlib.Path | str
    :returns: a set of pattern filenames.
    :rtype: set[str]
    """
    mapping_pattern_names = set(filtered_df.pattern.values)
    upheno_pattern_definition_paths = iter_upheno_pattern_definition_paths(upheno_dir)
    normalized_pattern_names = iter_normalized_pattern_names(
        upheno_pattern_definition_paths
    )
    pattern_names_in_mapping = set(
        name for name in normalized_pattern_names if name in mapping_pattern_names
    )
    pattern_file_names = set(
        lowercase_to_camel_case(name) for name in pattern_names_in_mapping
    )
    return pattern_file_names


def update_phipo_patterns(
    phipo_dir: Path | str,
    upheno_dir: Path | str,
    mapping_path: Path | str,
    robot_path: Path | str,
    id_label_mapping_dir: Path | str,
) -> None:
    """
    Update the pattern data files and the ``external.txt`` file in the PHIPO repository.

    :param args: parsed arguments from :func:`parse_args`.
    :type args: argparse.Namespace
    """
    phipo_dir = Path(phipo_dir)
    pattern_data_dir = phipo_dir / 'src' / 'patterns' / 'data' / 'default'
    external_txt_path = (
        phipo_dir / 'src' / 'patterns' / 'dosdp-patterns' / 'external.txt'
    )

    mapping_df = load_pattern_mapping_file(mapping_path)
    filtered_df = filter_mapping_df(mapping_df)

    pattern_file_names = get_pattern_file_names(filtered_df, upheno_dir)
    pattern_definition_file_names = set(name + '.yaml' for name in pattern_file_names)

    patterns_info = get_patterns_info(upheno_dir, pattern_definition_file_names)
    mapping_data_df = make_mapping_data_df(filtered_df)
    write_ontology_term_labels(phipo_dir, id_label_mapping_dir, robot_path)

    term_file_paths = Path(id_label_mapping_dir).glob('*.csv')
    id_label_mapping = load_id_label_mapping(term_file_paths)

    # Add an unmapped ID that wasn't exported for some reason
    id_label_mapping['CHEBI:26523'] = 'reactive oxygen species'

    data_file_dataframes = make_data_file_dataframes(
        mapping_data_df, patterns_info, id_label_mapping
    )
    validate_data_file_labels(data_file_dataframes)
    update_pattern_data_files(data_file_dataframes, pattern_data_dir)

    pattern_filenames = data_file_dataframes.keys()
    update_external_txt(external_txt_path, pattern_filenames, upheno_dir)
