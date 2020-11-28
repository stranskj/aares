"""
Angular reduction


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import h5z
import h5py
import numpy
import math
import ares.power as pwr

def create_bins(qmin, qmax, bins):
    '''
    Creates list of bins
    :param qmin: float
    :param qmax: float
    :param bins: int
    :return:
    '''

    out = []
    step = math.fabs(qmin-qmax)/bins

    for i in range(bins):
        out.append((qmin+i*step, qmin+(i+1)*step))

    return out

def integration_mask(bin, q_array,  frame_mask=None):
    '''
    Creates integration mask for a single bin
    :param q_array: numpy.array
    :param bin: tuple
    :param frame_mask:
    :return:
    '''

    bin = sorted(bin)

    q_mask = numpy.logical_and(bin[0] <= q_array, q_array < bin[1])

    if frame_mask is None:
        frame_mask = True

    return numpy.logical_and(q_mask, frame_mask)

def list_integration_masks(q_bins, q_array, frame_mask=None):
    '''
    Returns list of masks to be used for the integration.

    :param q_array: Numpy array of the frame shape with q-values of the
    :param q_bins: numpy.array of pairs
    :return:
    '''

    no_bins = len(q_bins)

    q_masks = map(integration_mask, q_bins, [q_array]*no_bins, [frame_mask]*no_bins)

    return list(q_masks)

def integrate(frame_arr, bin_masks):
    '''
    Calculate averages and stedevs across frames in all bins
    :param frame_arr: data; 3d np.array
    :param bin_masks: bin masks
    :return: np.array, np. array: averges and stdevs
    '''

    no_frame = frame_arr.shape[0]

    averages = []
    stdev    = []

    for binm in bin_masks:
        int_mask = numpy.array([binm]*no_frame)

        averages.append(numpy.average(frame_arr[int_mask]))
        stdev.append(numpy.std(frame_arr[int_mask]))

    #int_masks = [numpy.array([msk]*no_frame) for msk in bin_masks]

    #averages = [numpy.average(frame_arr[binm]) for binm in int_masks]
    #stdev    = [numpy.std(frame_arr[binm]) for binm in int_masks]

#    averages = map(numpy.average, [frame_arr]*len(bin_masks), int_masks)
#    stdev = map(numpy.std,  [frame_arr] * len(bin_masks), int_masks)

    return numpy.array(averages), numpy.array(stdev)

def test():
    import ares.q_transformation as qt
    import time
    import ares.mask as mask

    fin = '../data/10x60s_363mm_010Frames.h5'

    h5hd = h5z.SaxspointH5(fin)

    t0 = time.time()
    arrQ = qt.transform_detector_radial_q(h5hd)
    print(time.time() -t0)
    q_bins = create_bins(arrQ.min(), arrQ.max(), 750)
    print(time.time() - t0)
    q_masks = list_integration_masks(q_bins,arrQ)
    print(time.time() - t0)

    with h5z.FileH5Z(fin) as h5f:
        avr, std = integrate(h5f['entry/data/data'][:], q_masks)
    print(time.time() - t0)


 #   mask.draw_mask(q_masks[10],'q_mask10.png')

    with h5z.FileH5Z(fin) as fid:
        pass


def main():
    test()

if __name__ == '__main__':
    main()