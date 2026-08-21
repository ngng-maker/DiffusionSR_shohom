''' Helper functions for dataset processing. '''


def filter_data(array, thresholds, field_idxs, field_names):
    """Select requested fields and clip extreme values.

    Arrays are expected to store fields on the final axis when they have a
    field axis, e.g. ``(H, W, C)`` for 2D fields or ``(D, H, W, C)`` for true
    3D volumes. Single-field 2D arrays may be ``(H, W)``.
    """
    if len(array.shape) <= 2:
        if len(field_names) > 1:
            raise ValueError(
                f"Requested fields {field_names}, but array shape {array.shape} has no field axis. "
                "Regenerate a multifield split with channels on the final axis."
            )
        array = array.copy()
        for field in field_names:
            threshold = thresholds[field]
            array[array > threshold] = threshold
            array[array < -threshold] = -threshold
        return array

    missing_idxs = [idx for idx in field_idxs if idx >= array.shape[-1]]
    if missing_idxs:
        raise ValueError(
            f"Requested field indices {field_idxs}, but array shape {array.shape} only has "
            f"{array.shape[-1]} channel(s). Regenerate a split containing fields {field_names}."
        )

    selected = array[..., field_idxs].copy()
    if selected.ndim == array.ndim - 1:
        selected = selected[..., None]

    for selected_idx, field in enumerate(field_names):
        threshold = thresholds[field]
        channel = selected[..., selected_idx]
        channel[channel > threshold] = threshold
        channel[channel < -threshold] = -threshold
        selected[..., selected_idx] = channel
    return selected
