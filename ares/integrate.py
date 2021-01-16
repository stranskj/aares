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
import concurrent.futures
import os

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
    :return: np.array, np. array, np.array: averages, stddevs, number of points in bin
    '''

    no_frame = frame_arr.shape[0]

    averages = []
    stdev    = []
    num      = []

    for binm in bin_masks:
     #   int_mask = numpy.array([binm]*no_frame)
        binval = frame_arr[:,binm]
        if binval.size <= 0:
            averages.append(numpy.nan)
            stdev.append(numpy.nan)
        else:
            averages.append(numpy.nanmean(binval,dtype='float64'))
          #  stdev.append(numpy.nanstd(binval,dtype='float64')/math.sqrt(binval.size))
            stdev.append(numpy.sqrt(averages[-1])/math.sqrt(binval.size))
        num.append(binval.size)

    #int_masks = [numpy.array([msk]*no_frame) for msk in bin_masks]

    #averages = [numpy.average(frame_arr[binm]) for binm in int_masks]
    #stdev    = [numpy.std(frame_arr[binm]) for binm in int_masks]

#    averages = map(numpy.average, [frame_arr]*len(bin_masks), int_masks)
#    stdev = map(numpy.std,  [frame_arr] * len(bin_masks), int_masks)

    return numpy.array(averages), numpy.array(stdev), numpy.array(num)



def integrate_mp(frame_arr, bin_masks, nproc=None):
    '''
    Calculate averages and stddevs across frames in all bins, parallel in multiple chunks
    :param frame_arr: data; 3d np.array
    :param bin_masks: bin masks
    :return: np.array, np. array: averges and stdevs
    '''

    if nproc is None:
        nproc = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(nproc) as ex:
        results = ex.map(integrate, [frame_arr]*nproc, pwr.chunks(bin_masks,int(len(bin_masks)/nproc)+1))

        res = numpy.concatenate(list(results),axis=1)

        averages = res[0,:]
        stdev    = res[1,:]
        num      = res[2,:]

    return averages, stdev, num



def test():
    import ares.q_transformation as qt
    import time
    import ares.mask as am
    import ares.mask as mask

   # fin = '../data/10x60s_826mm_010Frames.h5'
    fin = '../data/W_826mm_005Frames.h5z'
   # fin = '../data/AgBeh_826mm.h5z'
    h5hd = h5z.SaxspointH5(fin)

    frame_mask = numpy.logical_and(am.read_mask_from_image('frame_alpha_mask.png',channel='A',invert=True),
                                   am.detector_chip_mask(det_type='Eiger R 1M'))

    t0 = time.time()
    arrQ = qt.transform_detector_radial_q(h5hd)
    print(time.time() -t0)
    q_bins = create_bins(arrQ.min(), arrQ.max(), 750)
    q_vals = qt.get_q_axis(q_bins)
    print(time.time() - t0)
    q_masks = list_integration_masks(q_bins,arrQ, frame_mask=frame_mask)
    print(time.time() - t0)

    with h5z.FileH5Z(fin) as h5f:
        frames = h5f['entry/data/data'][:]
        avr, std, num = integrate_mp(frames, q_masks)
    print(time.time() - t0)

    with open('data_826.dat','w') as fout:
        for q, I, s in zip(q_vals,avr, std):
            fout.write('{},{},{}\n'.format(q,I,s))

    import ares.statistics as stats
    import ares.draw2d as draw2d

    q_averages = stats.averages_to_frame_bins(q_masks, avr)
    q_stdevs = stats.averages_to_frame_bins(q_masks, std)
    draw2d.draw(q_averages,'frame_averages.png',Imax=1)

    relative_dev_pix = stats.local_relative_deviation(frames[0], q_averages, window=15)
    draw2d.draw(relative_dev_pix,'pix_dev.png', Imax=3, Imin=-3, cmap='PiYG')


 #   mask.draw_mask(q_masks[10],'q_mask10.png')

    with h5z.FileH5Z(fin) as fid:
        pass


def main():
    test()

if __name__ == '__main__':
    main()