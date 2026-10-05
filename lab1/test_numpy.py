import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    ChessInput,
)

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    chess,
)


# 1. sum_prod testi
matrices = np.array([
    [[1, 0], [0, 1]],
    [[2, 0], [0, 2]],
])

vectors = np.array([
    [[1], [2]],
    [[3], [4]],
])

print("sum_prod:")
print(sum_prod(MatrixVectorBatchInput(
    matrices=matrices,
    vectors=vectors
)))


# 2. binarize testi
matrix = np.array([
    [1, 5, 3],
    [7, 2, 4]
])

print("\nbinarize:")
print(binarize(BinarizeInput(
    matrix=matrix,
    threshold=3
)))


# 3. unique_rows / unique_columns
matrix2 = np.array([
    [1, 1, 2],
    [2, 3, 3],
])

print("\nunique_rows:")
print(unique_rows(MatrixInput(matrix=matrix2)))

print("\nunique_columns:")
print(unique_columns(MatrixInput(matrix=matrix2)))


# 5. chess
print("\nchess:")
print(chess(ChessInput(
    rows=3,
    columns=4,
    first=0,
    second=1
)))