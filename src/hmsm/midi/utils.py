from typing import Optional

import numpy as np


def preprocess_hupfeld_pedal(
    pedal_notes: np.array, threshold: Optional[int] = None
) -> np.array:
    """Preprocess pedal information on rolls with Hupfeld type pedal control

    Args:
        pedal_notes (np.array): Array containing data on the holes for the pedal track
        threshold (Optional[int], optional): Threshold to use when merging close pedal notes. If missing will use 1.8 times the height of the smallest hole in the input. Defaults to None.

    Returns:
        np.array: Array of the same shape as input with pedal information preprocessed
    """
    if threshold is None:
        threshold = 1.8 * pedal_notes[:, 1].min()

    dists = np.diff(pedal_notes[:, 0], append=0) - pedal_notes[:, 1]

    rows = list()

    current_row = pedal_notes[0, :]

    for i in range(1, len(pedal_notes)):
        if dists[i - 1] < threshold:
            current_row[1] = (pedal_notes[i, 0] + pedal_notes[i, 1]) - current_row[0]
        else:
            rows.append(current_row)
            current_row = pedal_notes[i, :]

    rows.append(current_row)

    return np.vstack(rows)
