import datetime

import h5z, h5py
import aares

def Reduced1D_factory(base_class=h5z.SaxspointH5):

    assert issubclass(base_class, h5z.InstrumentFileH5)
    def __init__(self, path):
        super(type(self), self).__init__(path)
        self.attrs['reduced'] = True
        self.attrs['HDF5_Version'] = h5z.hdf5_version
        self.attrs['creator'] = 'AAres {}'.format(aares.version)
        self.attrs['file_time'] = datetime.datetime.now().isoformat()

    @staticmethod
    def is_type(val):

        attributes = {}
        if h5z.is_h5_file(val):
            with  h5z.FileH5Z(val, 'r') as fin:
                attributes.update(fin.attrs)
        elif isinstance(val, h5z.GroupH5) or isinstance(val, h5py.Group):
            attributes.update(val.attrs)
        else:
            raise TypeError('Input should be h5py.Group-like object.')
        try:
            out  = attributes['reduced']
        except KeyError:
            out = False

        return out
    @property
    def q_values(self):
        return self['entry/processed/q_values']

    @q_values.setter
    def q_values(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='q_values')
        dts.attrs['long_name'] = 'Coordinate in reciprocal space'
        dts.attrs['units'] = 'Unknown'
        self['entry/processed/q_values'] = dts

    @property
    def q_value_units(self):
        return self['entry/processed/q_values'].attrs['units']

    @q_value_units.setter
    def q_value_units(self, val):
        if val not in ['1/nm', '1/A']:
            raise ValueError('Invalid unit type. Allowd: 1/nm, 1/A.  Given: {}'.format(val))
        self['entry/processed/q_values'].attrs['units'] = val

    @property
    def intensity(self):
        return self['entry/processed/intensity']

    @intensity.setter
    def intensity(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='intensity')
        dts.attrs['long_name'] = 'Scattering intensity'
        dts.attrs['units'] = 'a.u.'
        self['entry/processed/intensity'] = dts


    @property
    def intensity_units(self):
        return self['entry/processed/intensity'].attrs['units']

    @intensity_units.setter
    def intensity_units(self, val):
#        if val not in ['1/nm',  '1/A']:
#            raise ValueError('Invalid unit type. Allowd: 1/nm, 1/A.  Given: {}'.format(val))
        self['entry/processed/intensity'].attrs['units'] = val

    @property
    def intensity_sigma(self):
        return self['entry/processed/intensity_sigma']

    @intensity_sigma.setter
    def intensity_sigma(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='intensity_sigma')
        dts.attrs['long_name'] = 'Sigma error of scattering intensity'
        dts.attrs['units'] = 'a.u.'
        self['entry/processed/intensity_sigma'] = dts

    @property
    def redundancy(self):
        return self['entry/processed/redundancy']

    @redundancy.setter
    def redundancy(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='redundancy')
        dts.attrs['long_name'] = 'Number of pixels in q-bin'
        dts.attrs['units'] = 'pixels'
        self['entry/processed/redundancy'] = dts


    cls_1D = type("Reduced1D", (base_class,),
                  {
                      "__init__": __init__,
                      "is_type":  is_type,
                      "q_values": q_values,
                      "q_value_units": q_value_units,
                      "intensity": intensity,
                      "intensity_units": intensity_units,
                      "intensity_sigma": intensity_sigma,
                      "redundancy": redundancy,
                  })
    return cls_1D


def test_Reduced1D():
    fin = "../data/AgBeh_826mm.h5z"
    header = h5z.SaxspointH5(fin)
    reduced1d = Reduced1D_factory(type(header))
    hd1 = reduced1d(header._h5)

    import numpy
    arr = numpy.array(range(10))
    hd1.intensity = arr
    hd1.intensity_sigma = arr
    hd1.q_values = arr
    hd1.redundancy = arr
    hd1.q_value_units = "1/A"
    hd1.intensity_units= "cm^-2"
    hd1.intensity
    hd1.intensity_sigma
    hd1.q_values
    hd1.redundancy
    hd1.q_value_units
    hd1.intensity_units
    hd1.write('AgBeh_826mm_reduced.h5')

    assert reduced1d.is_type('AgBeh_826mm_reduced.h5')
