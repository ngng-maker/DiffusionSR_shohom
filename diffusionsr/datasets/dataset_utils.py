''' Helper functions for dataset processing. '''


def filter_data(array, thresholds, field_idxs, field_names):
    '''
    Filters out extreme values in the data due to numerical noise, according to thresholds.

    Parameters:
        array (numpy.ndarray): The data array to be filtered. Can be 2D or 3D, where the last dimension represents channels or fields.
        thresholds (dict): A dictionary mapping field names to their respective threshold values.
        field_idxs (list of int): Indices of the fields to filter within the array's last dimension.
        field_names (list of str): Names of the fields corresponding to the indices in `field_idxs`.

    Returns:
        numpy.ndarray: The filtered data array with extreme values clipped to the specified thresholds.

    Notes:
        - For each field specified in `field_idxs` and `field_names`, values greater than the positive threshold are clipped to the threshold, 
          and values less than the negative threshold are clipped to the negative threshold.
        - The function supports both 2D and 3D arrays. In 3D arrays, the filtering is applied to each channel independently.

    Example:
        >>> import numpy as np
        >>> array = np.random.randn(5, 5, 3)
        >>> thresholds = {'temperature': 10, 'velocity': 5}
        >>> field_idxs = [0, 2]
        >>> field_names = ['temperature', 'velocity']
        >>> filtered_array = filter_data(array, thresholds, field_idxs, field_names)
    '''
  
    if len(array.shape) > 2:
        for i, field in zip(field_idxs, field_names):
            threshold = thresholds[field]
            array_channel = array[:,:, i]
            array_channel[array_channel > threshold] = threshold
            array_channel[array_channel < -threshold] = -threshold
            array[:, :, i] = array_channel
            array[:,:, i] = array_channel
    else:
        for i, field in zip(field_idxs, field_names):
            threshold = thresholds[field]
            array[array > threshold] = threshold
            array[array < -threshold] = -threshold

    return array