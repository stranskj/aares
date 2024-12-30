import pytest

import h5z

@pytest.mark.parametrize('fin',[('../data/LVA_2024-12_LA1.h5z')])
def test_SaxspointH5(fin):

    assert h5z.is_h5_file(fin)

    header = h5z.SaxspointH5(fin)

    assert isinstance(header.sdd, float)