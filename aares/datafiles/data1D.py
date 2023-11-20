import h5z


def Reduced1D_factory(base_class=h5z.SaxspointH5):
    def __init__(self, path):
        super(self).__init__(path)

    cls_1D = type("Reduced1D", (base_class,),
                  {
                      "__init__": __init__,
                  })
    return cls_1D


def test_Reduced1D():
    fin = "../data/AgBeh_826mm.h5z"
    header = h5z.SaxspointH5(fin)
    reduced = Reduced1D_factory(type(header))

    assert isinstance(reduced, "Reduced1D")