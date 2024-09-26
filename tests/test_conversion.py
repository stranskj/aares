import h5z
import aares.q_transformation as qtr


def test_write():
    header = h5z.SaxspointH5('../data/10x60s_826mm_010Frames.h5')
    scope = qtr.extract_geometry_to_phil(header)
    print(scope.as_str(attributes_level=2))
    pass

#if __name__ == "__main__":
#    test()