import h5z
import aares.q_transformation as qtr
import copy


def test_write():
    header = h5z.SaxspointH5('../data/AgBeh_826mm.h5z')
    scope = qtr.extract_geometry_to_phil(header)
    print(scope.as_str(attributes_level=2))
    pass

def test_h5z_beamXY():
    header = h5z.SaxspointH5('../data/AgBeh_826mm.h5z')
    header_copy = copy.deepcopy(header)
    header_copy.beam_center_px = 600, 300
    assert header_copy.beam_center_px[0] == 600,\
        header_copy.beam_center_px[1] == 300

#if __name__ == "__main__":
#    test()