"""Grades results.csv against truth.json. Usage: python grade.py results.csv truth.json

Exits 0 if everything passes, 1 otherwise, and prints why.
"""
import json
import math
import sys

import pandas as pd

COLUMNS = ['month', 'mean_savi', 'valid_fraction']


def grade(results_path, truth_path):
    """Returns a list of problems, an empty list means a pass."""
    with open(truth_path) as f:
        truth = json.load(f)

    try:
        table = pd.read_csv(results_path, dtype={'month': str})
    except (OSError, ValueError) as error:
        return [f'could not read results: {error}']

    if list(table.columns) != COLUMNS:
        return [f'columns should be {COLUMNS}, found {list(table.columns)}']
    expected = [row['month'] for row in truth['months']]
    if table['month'].tolist() != expected:
        return ['months should be every YYYY-MM from oldest to newest, one row each']

    problems = []
    for found, row in zip(table.to_dict('records'), truth['months']):
        for column, tolerance in truth['tolerance'].items():
            value = found[column]
            if not math.isfinite(value) or abs(value - row[column]) > tolerance:
                problems.append(f"{row['month']} {column}: found {value}, expected {row[column]:.4f} +/- {tolerance}")
    return problems


if __name__ == '__main__':
    found_problems = grade(sys.argv[1], sys.argv[2])
    for problem in found_problems[:10]:
        print(f'FAIL {problem}')
    if found_problems:
        print(f'{len(found_problems)} problem(s) in total')
        sys.exit(1)
    print('PASS')
