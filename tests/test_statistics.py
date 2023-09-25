from aares.statistics import *

def test_statistics():

    #fin = '../data/10x60s_826mm_010Frames.h5'
    #fin = '../data/buffer_16x60.h5z'
    fin = '../data/W_826mm_005Frames.h5z'
    #fin = '../data/AgBeh_826mm.h5z'

    h5f = h5z.SaxspointH5(fin)
    frames = h5f.data[:]

 #   paded = numpy.pad(frames[0],pad_width=2, mode='constant',constant_values=-2)
 #   paded[paded<0]= numpy.nan
  #  slwnd = rolling_window(paded,(5,5))

   # avr = numpy.average(slwnd,axis=(2,3))

 #   avr = sliding_average_frame(frames[0],window=5)
    import aares.integrate
    import aares.q_transformation
    print('Reading...')
    arrQ = aares.q_transformation.transform_detector_radial_q(h5f)
    qvals, qbins = aares.integrate.prepare_bins(arrQ,qmin=0.1,qmax=2)
    qbins_obj = aares.integrate.ReductionBins()
    qbins_obj.create_bins(arrQ)
    nums=redundancy_in_bin(qbins_obj)
    print('Getting CC12')
    cc12_many = []
    cc12_w_many = []
    for i in range(10):
        cc12, cc12_w = set_cc12(frames[0],qbins,10)
        cc12_many.append(cc12)
        cc12_w_many.append(cc12_w)

    cc12_a = numpy.nanmean(cc12_many)
    cc12_w_a = numpy.nanmean(cc12_w_many,axis=0)
    cc12_w_s = numpy.nanstd(cc12_w_many, axis=0)

    with open('cc12.dat','w') as fout:
        for q, cc, s in zip(numpy.array_split(qvals,10), cc12_w_a, cc12_w_s):
            fout.write('{} {} {}\n'.format(q[-1], cc,s))
    print(cc12)
