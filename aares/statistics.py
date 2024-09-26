"""
Data statistics


@author:     Jan Stransky

@copyright:  2019, 2020, 2021 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import aares.export
import h5z
import h5py
import numpy
import math
import aares.power as pwr
import concurrent.futures
import os

def averages_to_frame_bins(q_bins, averages):
    '''
    Returns frame filled with averages from the angular reduction assigned back to the individual pixels

    :param q_bins: Q-bin masks used for the angular reduction
    :param averages: Averages
    :return: Averages assigned to the pixels
    '''

    out_array = numpy.empty(q_bins[0].shape,dtype=float)
    out_array[:] = numpy.nan

    for bin, avr in zip(q_bins,averages):
        out_array[bin] = avr

    return out_array

def local_relative_deviation(frame, averages, window):
    '''
    Calculates relative deviation of sliding average with box size of "window"
    :param frame:
    :param averages:
    :param window:
    :return:
    '''

    sliding_avr = sliding_average_frame(frame, window)

    return (sliding_avr - averages) / averages

def relative_pixel_deviation(frame, averages):
    '''
    Calculate relative deviation of individual pixels from the average

    :param frame: Original fram
    :param averages: Averages assigned to the pixel
    :return: Relative deviation of the pixels
    '''

    assert frame.shape == averages.shape

    return (frame - averages) / averages

def sliding_average_frame(frame, window):
    '''
    Calculates sliding average of the frame pixels. The output array is of the dimension of the frame.
    :param frame:
    :param window: width of the averiging window. Window of a square dimensions is used. Should be an even number
    :return:
    '''

    assert window % 2 == 1

    padding = int(window /2)

    paded = numpy.pad(frame.astype(float),pad_width=padding, mode='constant',constant_values=numpy.nan)
    paded[paded<0]= numpy.nan

    arrays_window = rolling_window(paded,(window,window))

    avr = numpy.nanmean(arrays_window,axis=(2,3))
    return avr

def set_cc12(dataset, qbins, wedges=None):
    '''
    Returns CC1/2 for the dataset

    :param dataset: Single frame to be analyzed
    :type dataset: numpy.array
    :param qbins: q-bin masks
    :type qbins: list of numpy.arrays
    :param wedges: Number of q-range wedges, in which cc12 is also calculated
    :return: float
    '''

    half1 = []
    half2 = []

    def bin_worker(bin):
        h1 = []
        h2 = []

        split = random_sets(dataset[bin],2)
        if (len(split[0]) > 0) and (len(split[1]) > 0):
            h1.append(numpy.nanmean(split[0]))
            h2.append(numpy.nanmean(split[1]))
        else:
            h1.append(numpy.nan)
            h2.append(numpy.nan)

        return h1, h2

    halfs = numpy.array(pwr.map_th(bin_worker, qbins))
    half1 = halfs[:,0,0]
    half2 = halfs[:,1,0]

    cc12 = numpy.corrcoef(numpy.array(half1),numpy.array(half2))[0,1]

    if wedges is not None:
        cc12_wedges = []
        for h1, h2 in zip(numpy.array_split(half1,wedges),
                          numpy.array_split(half2,wedges)):
            cc12_wedges.append(numpy.corrcoef(numpy.array(h1),numpy.array(h2))[0,1])
    else:
        cc12_wedges = None
    return cc12, cc12_wedges

def random_sets(dataset, nsets):
    '''
    Returns random split of the dataset

    :param dataset:
    :param nsets: number of sets to the dataset to be divided into
    :return: list of bool arrays
    '''
    indices = numpy.random.permutation(dataset.flatten())
    chunks = numpy.array_split(indices,nsets)

    subset_masks = []

  #  for chnk in chunks:
   #     arr = numpy.full(dataset.shape, False)
    #    arr[chnk] = True
     #   subset_masks.append(arr)

    return chunks


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    (from https://gist.github.com/seberg/3866040)

    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.

    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.

    Examples
    --------
    >>> a = numpy.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])

    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])

    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)

    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])

    Or delay embedding (3D embedding with delay 2):
    >>> x = numpy.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = numpy.asarray(array)
    orig_shape = numpy.asarray(array.shape)
    window = numpy.atleast_1d(window).astype(int) # maybe crude to cast to int...

    if axes is not None:
        axes = numpy.atleast_1d(axes)
        w = numpy.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if numpy.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = numpy.ones_like(orig_shape)
    if asteps is not None:
        asteps = numpy.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps

        if numpy.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = numpy.ones_like(window)
    if wsteps is not None:
        wsteps = numpy.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if numpy.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if numpy.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = numpy.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = numpy.concatenate((shape, window))
        new_strides = numpy.concatenate((strides, new_strides))
    else:
        _ = numpy.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = numpy.zeros(len(shape)*2, dtype=int)
        new_strides = numpy.zeros(len(shape)*2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return numpy.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

def redundancy_in_bin(bins):
    '''Returns number of pixels in each bin'''
   # assert type(bins, aares.integrate.ReductionBins)
    return [bn.sum() for bn in bins.bin_masks]

def export_redundancy(bins, fiout, separator=" "):
    '''Writes redundancy'''
    reduncies = redundancy_in_bin(bins)
    masked = len(reduncies)*['nan']

    aares.export.write_atsas(bins.q_axis, reduncies, masked, fiout, header=['Number of pixels in indvidual q-shells',
                                                                            'q{sep}used{sep}masked'.format(sep=separator)])

