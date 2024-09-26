
def test_mp_worker():
    import h5z
    from aares.power import mp_worker
    out = mp_worker(h5z.SaxspointH5, '../data/AgBeh_363mm.h5z')

    assert isinstance(out, h5z.SaxspointH5)