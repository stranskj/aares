import pytest
import freephil as phil

from aares import datafiles

@pytest.mark.parametrize(('file_name', 'pattern'),[
    ('0_buffer25mg_AD6', 'buffer'),
    ('0_buffer25mg_AD6', ['buffer', 'matrix']),
    ('123_LD6_LD6_Temp', 'L.6'),
    ('123_LD6_LD6_Temp', 'L.[0-9]'),
])
def test_is_background(file_name, pattern):
    assert datafiles.is_background(file_name, pattern)

def test_detect_background():
    group_scope = phil.parse('''
    file.name=0_sgfkjbkjnbs_54mm
    file.name=1_dhslkhg_28mm
    file.name=2_buffer_sgjdslkjh_28mm
    ''')
    group_obj = datafiles.group_phil.fetch(group_scope).extract()
    datafiles.detect_background(group_obj,'buffer')
    assert len([fi for fi in group_obj.file if fi.is_background])==1