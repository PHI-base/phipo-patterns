# PHIPO phenotype pattern maker

Python package to maintain a mapping between phenotype terms (ontology classes) in the [Pathogen–Host Interaction Phenotype Ontology](https://github.com/PHI-base/phipo) (PHIPO) and the phenotype design patterns used by the [Unified Phenotype Ontology](https://github.com/obophenotype/upheno) (uPheno).

The mapping information is stored in the PHIPO pattern mapping table, which is internally maintained as a spreadsheet by the PHI-base team. This package processes an export of that spreadsheet as a CSV file.

An example of the columns used in the pattern mapping table can be seen in the file `pattern_mapping_table_example.csv` in this repository.

## Installation

Install the latest release from GitHub:

```
python -m pip install 'phipo_patterns@git+https://github.com/PHI-base/phipo-patterns.git@0.2.0'
```

Or install the latest commit on the `main` branch:

```
python -m pip install 'phipo_patterns@git+https://github.com/PHI-base/phipo-patterns.git@main'
```

## Usage

The `phipo_patterns` package provides a command-line interface with multiple commands. Run the following in a terminal for help:

```
python -m phipo_patterns --help
```

### `make_patterns`

Create pattern data files from the PHIPO pattern mapping table and update the `external.txt` file in the PHIPO repository folder.

- `--mapping-file`: path to the PHIPO pattern mapping table. The table should be in CSV file format.
- `--ontology-dir`: path to the PHIPO repository on your local filesystem. This repository can be cloned from [PHI-base/phipo](https://github.com/PHI-base/phipo).
- `--upheno-dir`: path to the uPheno repository on your local filesystem. This repository can be cloned from [obophenotype/uhpeno](https://github.com/obophenotype/upheno).
- `--robot-path`: path to the robot.jar file on your local filesystem. See the [ROBOT website](https://robot.obolibrary.org/) for instructions on installing ROBOT.


The pattern data files are TSV files that contain the term ID and labels for the PHIPO term(s) to which the phenotype pattern should be applied, plus the term ID and labels for the variables used by the pattern. The pattern data files are located in the following folder in the PHIPO repository:

phipo > src > patterns > data > default

Below is an example of a row in the pattern mapping table (most columns are not shown for clarity).

| label | term | pattern | variables |
| --- | --- | --- | --- |
| asexual spores absent | PHIPO:0000061 | abnormal absence of anatomical entity | CL:0000605 |

This row will be converted to the following row in the pattern data file:

| defined_class | defined_class_label | anatomical_entity | anatomical_entity_label
| --- | --- | --- | --- |
| PHIPO:0000061 | asexual spores absent | CL:0000605 | fungal asexual spore |

The ROBOT tool is used to extract term labels from mirrored ontology files in the local PHIPO repository folder. These labels are used to populate the label columns in the pattern data files. 

### `sync_mapping`

Synchronize the PHIPO pattern mapping table with the latest version of PHIPO and the latest pattern names from uPheno. Export the updated pattern mapping table as a CSV file.

- `--mapping-file`: path to the PHIPO pattern mapping table. The table should be in CSV file format.
- `--ontology-dir`: path to the PHIPO repository on your local filesystem. This repository can be cloned from [PHI-base/phipo](https://github.com/PHI-base/phipo).
- `--upheno-dir`: path to the uPheno repository on your local filesystem. This repository can be cloned from [obophenotype/uhpeno](https://github.com/obophenotype/upheno).
- `--output`: output path for the updated PHIPO pattern mapping table.

The ROBOT tool is used to extract the term ID, label, definition, exact synonym, subset, and OBO namespace for each term in PHIPO. These new values are merged in to the pattern mapping table.

The **error** column is automatically updated to mark terms that are obsolete, merged into another term, or have invalid pattern names or variables.

Terms whose labels have changed since the last synchronization are marked as modified in the **modified** column.

## License

The `phipo-patterns` project is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
