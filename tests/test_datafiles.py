import pytest

import h5z
import aares.datafiles
#import aares.datafiles.data1D
def test_Reduced1D_write():
    fin = "../data/AgBeh_826mm.h5z"
    header = h5z.SaxspointH5(fin)
    #reduced1d = aares.datafiles.data1D.Reduced1D_factory(type(header))
    #hd1 = reduced1d(header._h5)
    hd1 = aares.datafiles.Reduced1D(header)

    import numpy
    arr = numpy.array(range(12))
    hd1.parents = [fin, header]
    hd1.intensity = arr
    hd1.intensity
    hd1.intensity_sigma = arr
    hd1.q_values = arr
    hd1.redundancy = arr
    hd1.q_value_units = "1/angstrom"
    hd1.intensity_units= "1/cm"
    hd1.scale = 45.4678

    hd1.intensity
    hd1.intensity_sigma
    hd1.q_values
    hd1.redundancy
    hd1.q_value_units
    hd1.intensity_units
    hd1.scale
    hd1.parents
    hd1.write('AgBeh_826mm_reduced.h5')

    assert hd1.is_type('AgBeh_826mm_reduced.h5')

def test_Subtract1D_write():
    fin = "../data/AgBeh_826mm.h5z"
    header = h5z.SaxspointH5(fin)
    hd1 = aares.datafiles.Subtract1D(header)
    import numpy
    arr = numpy.array(range(12))
    hd1.intensity = arr
    hd1.intensity_sigma = arr
    hd1.q_values = arr
    hd1.update_attributes()

    assert hd1['entry/data/I'][2] == 2
    hd1.write('AgBeh_826mm_subtracted.h5')
    assert hd1.is_type('AgBeh_826mm_subtracted.h5')




def test_Reduced1D_read():

    assert aares.datafiles.data1D.Reduced1D.is_type('AgBeh_826mm_reduced.h5')

#    file_type = aares.datafiles.get_file_type('')

def test_get_file_type():
    file_class = aares.datafiles.get_file_type('../data/AgBeh_826mm.h5z')
    assert file_class == h5z.SaxspointH5

    file_class = aares.datafiles.get_file_type('AgBeh_826mm_reduced.h5')
    header = file_class('AgBeh_826mm_reduced.h5')
    assert header.attrs['reduced']

def test_read_file():
    header = aares.datafiles.read_file('AgBeh_826mm_reduced.h5')
    assert header.attrs['reduced']
    assert header.intensity[6] == 6

@pytest.mark.skip(reason="not implemented properly")
def test_Reduced1D_pickle():
    import pickle
    header = aares.datafiles.read_file('AgBeh_826mm_reduced.h5')
   # header = aares.datafiles.read_file('../data/AgBeh_826mm.h5z')
    pckl = pickle.dumps(header)
    unpck_header = pickle.loads(pckl)

    assert unpck_header.attrs['reduced']
    assert unpck_header.intensity[6] == 6


def test_get_headers_1D():
    import aares.power
    headers = aares.power.get_headers_dict(['AgBeh_826mm_reduced.h5'])#['reduced/1_SDD_SDD_793mm_010Frames.h5r', 'reduced/2_SDD.001_SDD.001_793mm_090Frames.h5r'])
    assert len(headers) == 1
