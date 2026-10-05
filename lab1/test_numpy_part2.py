import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from grader_contracts.numpy_tasks import (
    RandomMatrixInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)

from numpy_tasks import (
    matrix_statistics,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)


# -------------------------
# Görev 4: matrix_statistics
# -------------------------

stats = matrix_statistics(
    RandomMatrixInput(
        rows=2,
        columns=3,
        mean=0,
        std=1,
        seed=42,
    )
)

print("matrix:")
print(stats.matrix)

print("\nrow_means:")
print(stats.row_means)

print("\ncolumn_means:")
print(stats.column_means)

print("\nrow_variances:")
print(stats.row_variances)

print("\ncolumn_variances:")
print(stats.column_variances)


# -------------------------
# Görev 6: draw_rectangle
# -------------------------

rectangle = draw_rectangle(
    RectangleInput(
        width=4,
        height=2,
        image_height=6,
        image_width=8,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )
)

print("\nrectangle shape:")
print(rectangle.shape)

print("\nrectangle:")
print(rectangle)


# -------------------------
# Görev 6: draw_ellipse
# -------------------------

ellipse = draw_ellipse(
    EllipseInput(
        semi_axis_x=2,
        semi_axis_y=1,
        image_height=5,
        image_width=7,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0),
    )
)

print("\nellipse shape:")
print(ellipse.shape)

print("\nellipse:")
print(ellipse)


# -------------------------
# Görev 7: analyze_time_series
# -------------------------

series_stats = analyze_time_series(
    TimeSeriesInput(
        values=np.array([1, 3, 2, 5, 1, 4]),
        window=3,
    )
)

print("\ntime series mean:")
print(series_stats.mean)

print("\ntime series variance:")
print(series_stats.variance)

print("\ntime series std:")
print(series_stats.std)

print("\nlocal maxima indices:")
print(series_stats.local_maxima_indices)

print("\nlocal minima indices:")
print(series_stats.local_minima_indices)

print("\nmoving average:")
print(series_stats.moving_average)


# -------------------------
# Görev 8: one_hot
# -------------------------

encoded = one_hot(
    OneHotInput(
        labels=np.array([0, 2, 3, 0]),
        class_count=None,
    )
)

print("\none_hot:")
print(encoded)