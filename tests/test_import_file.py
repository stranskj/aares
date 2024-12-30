import pytest

from aares import import_file

@pytest.mark.parametrize(('file_name', 'pattern'),[
    ('0_buffer25mg_AD6', 'buffer'),
    ('0_buffer25mg_AD6', ['buffer', 'matrix']),
    ('123_LD6_LD6_Temp', 'L.6'),
    ('123_LD6_LD6_Temp', 'L.[0-9]'),
])
def test_is_background(file_name, pattern):
    assert import_file.is_background(file_name, pattern)