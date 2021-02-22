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
import os, logging
import freephil as phil

phil_core_str ='''
    bins_number = None
    .type = int
    .expert_level = 0
    .help = Number of bins to which data are devided. If 0 or None, determined automatically.
    
    q_range = 0 0
    .type = floats(2)
    .expert_level = 0
    .help = Q range to be used for the reduction
   
    beam_normalize
    .help = Adjust data for beam variation. Pick one of the options q_range or real_space. If 0, it is not used. 
    {
        q_range = 0 0
        .type = floats(2)
        .expert_level = 0
        .help = Adjust to the mean within the q-range. For example, it can be used for backgraound normalization.
        
        real_space = 0 0
        .type = floats(2)
        .expert_level = 0
        .help = Adjust to the mean within the real distance on the detector. Useful for normalization using semi-transparent beamstop: use balue slightly smaller than the beamstop size
        
        beamstop_transmission = 1
        .type = float
        .expert_level = 1
        .help = Transmission constant of the semitransparent beamstop.
    }
     
'''

phil_core = phil.parse(phil_core_str)

phil_job_core = phil.parse('''
    input.file_name = None
    .multiple = True
    .type = path
    
    reduction  {
    '''+ phil_core_str+'''
    }
''', process_includes=True)

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

    q_masks = pwr.map_th(integration_mask, q_bins, [q_array]*no_bins, [frame_mask]*no_bins)

    return list(q_masks)

def integrate(frame_arr, bin_masks):
    '''
    Calculate averages and stddevs across frames in all bins
    :param frame_arr: data; 3d np.array
    :param bin_masks: bin masks
    :return: np.array, np. array: averges and stdevs
    '''

    if isinstance(bin_masks, numpy.ndarray) and len(bin_masks.shape) == 2:
        bin_masks.shape = (1, *bin_masks.shape)

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
            stdev.append(numpy.nanstd(binval,dtype='float64')/math.sqrt(binval.size))
          #  stdev.append(numpy.sqrt(averages[-1])/math.sqrt(binval.size))
        num.append(binval.size)

    #int_masks = [numpy.array([msk]*no_frame) for msk in bin_masks]

    #averages = [numpy.average(frame_arr[binm]) for binm in int_masks]
    #stdev    = [numpy.std(frame_arr[binm]) for binm in int_masks]

#    averages = map(numpy.average, [frame_arr]*len(bin_masks), int_masks)
#    stdev = map(numpy.std,  [frame_arr] * len(bin_masks), int_masks)

    return numpy.array(averages), numpy.array(stdev), numpy.array(num)



def integrate_mp(frame_arr, bin_masks, nproc=None):
    '''
    Calculate averages and stedevs across frames in all bins, parallel in multiple chunks
    :param frame_arr: data; 3d np.array
    :param bin_masks: bin masks
    :return: np.array, np. array: averges and stdevs
    '''

    if nproc is None:
        nproc = os.cpu_count()

    if len(frame_arr.shape) == 2:
        frame_arr.shape = (1,*frame_arr.shape)

    with concurrent.futures.ThreadPoolExecutor(nproc) as ex:
        results = ex.map(integrate, [frame_arr]*nproc, pwr.chunks(bin_masks,int(len(bin_masks)/nproc)+1))
        #results = pwr.map_th(integrate, [frame_arr]*len(bin_masks), bin_masks, nchunks=nproc)

        res = numpy.concatenate(list(results),axis=1)

        averages = res[0,:]
        stdev    = res[1,:]
        num      = res[2,:]

    return averages, stdev, num

def prepare_bins(arrQ, qmin=None, qmax=None, bins=None, frame_mask=None):
    """
    High level function, which prepares q_bins and integration masks

    :param arrQ: Q-transformed frame e.g. array, which holds q-value for each pixel
    :type arrQ: np.array
    :param qmin: q_min value
    :type qmin: float
    :param qmax: q_max value
    :type qmax: float
    :param bins: number of bins
    :type bins: int
    :param frame_mask: Additional mask for the frame.
    :type frame_mask: np.array(bools)
    :return: Returns array of q_values, and corresponding list of masks used for the integration
    """

    import ares.q_transformation as qt

    if frame_mask is None:
        frame_mask = arrQ > 0

    if qmin is None:
        qmin = arrQ[frame_mask].min()
    if qmax is None:
        qmax = arrQ[frame_mask].max()

    if (bins is None) or (bins == 0):
        origin = numpy.unravel_index(numpy.argmin(arrQ, axis=None),arrQ.shape)
        to_corners = [math.sqrt((origin[0] - 0)**2 + (origin[1] - 0)**2),
                      math.sqrt((origin[0] - arrQ.shape[0])**2 + (origin[1] - 0)**2),
                      math.sqrt((origin[0] - 0)**2 + (origin[1] - arrQ.shape[1])**2),
                      math.sqrt((origin[0] - arrQ.shape[0])**2 + (origin[1] - arrQ.shape[1])**2)]
        bins = int(max(to_corners)*0.7)

    q_bins = create_bins(qmin,qmax, bins)
    q_masks =  list_integration_masks(q_bins, arrQ, frame_mask)
    return qt.get_q_axis(q_bins), q_masks

def test():
    import ares.q_transformation as qt
    import time
    import ares.mask as am
    import ares.mask as mask

    fin = '../data/10x60s_826mm_010Frames.h5'

    h5hd = h5z.SaxspointH5(fin)

    frame_mask = numpy.logical_and(am.read_mask_from_image('frame_alpha_mask_180.png',channel='A',invert=True),
                                   am.detector_chip_mask('Eiger R 1M'))

    t0 = time.time()
    arrQ = qt.transform_detector_radial_q(h5hd)
    print(time.time() -t0)
    q_bins = create_bins(arrQ.min(), arrQ.max(), 750)
    q_vals = qt.get_q_axis(q_bins)
    print(time.time() - t0)
    q_masks = list_integration_masks(q_bins,arrQ, frame_mask=frame_mask)
    print(time.time() - t0)

    with h5z.FileH5Z(fin) as h5f:
        avr, std, num = integrate_mp(h5f['entry/data/data'][:], q_masks)
    print(time.time() - t0)

    with open('data_826.dat','w') as fout:
        for q, I, s in zip(q_vals,avr, std):
            fout.write('{},{},{}\n'.format(q,I,s))



 #   mask.draw_mask(q_masks[10],'q_mask10.png')

    with h5z.FileH5Z(fin) as fid:
        pass


def main():
    test()

if __name__ == '__main__':
    main()