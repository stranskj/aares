import aares.subtract
from aares.datafiles import Reduced1D, Subtract1D

def test_subtract():
    data = Reduced1D('reduced/10_lys.h5r')
    buffer = Reduced1D('reduced/09_buffer.h5r')

    out = aares.subtract.subtract_reduced(data, buffer)

    assert out.intensity.shape == data.intensity.shape
    assert isinstance(out, Subtract1D)
    assert out['entry/data/I'][0] == data.intensity[0] - buffer.intensity[0]
    assert out.intensity[0] == data.intensity[0] - buffer.intensity[0]

    out.write('subtracted.h5s')

    assert Subtract1D.is_type('subtracted.h5s')

def test_read_subtracted():

    data = Subtract1D('subtracted.h5s')
    wlk = data._data1d.walk()
    assert '/entry/data/I' not in wlk

    assert data.intensity.shape == data.q_values.shape == data.redundancy.shape
    assert len(data.parents) == 2

