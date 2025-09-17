# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import sys

from phipo_patterns import cli

if __name__ == '__main__':
    # First item of args is the script name: skip it
    cli.run(sys.argv[1:])
