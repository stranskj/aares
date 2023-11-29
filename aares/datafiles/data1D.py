import datetime
import os.path

import h5z, h5py
import aares
import numpy
import logging

class Reduced1D_meta:
    @staticmethod
    def is_type(val):

        attributes = {}
        if h5z.is_h5_file(val):
            with h5z.FileH5Z(val, 'r') as fin:
                attributes.update(fin.attrs)
        elif isinstance(val, h5z.GroupH5) or isinstance(val, h5py.Group):
            attributes.update(val.attrs)
        else:
            raise TypeError('Input should be h5py.Group-like object.')
        try:
            out = attributes['reduced']
        except KeyError:
            out = False

        return out

    @staticmethod
    def base_class_name(fin_path):
        try:
            if h5z.is_h5_file(fin_path):
                with h5z.FileH5Z(fin_path,'r') as fin:
                    return fin.attrs['base_type']
            elif isinstance(fin_path, h5z.GroupH5) or isinstance(fin_path, h5py.Group):
                return fin_path.attrs['base_type']
        except KeyError:
            raise ValueError('This is not a Reduced1D file.')
        else:
            raise ValueError('This is not a Reduced1D file.')

def Reduced1D_factory(base_class=h5z.SaxspointH5):

    assert issubclass(base_class, h5z.InstrumentFileH5)
    def __init__(self, path):
        super(type(self), self).__init__(path)

#        self.skip_entries.append('entry/processed/intensity')
#         empty_arr = numpy.empty(0)
#         self.q_values = empty_arr
#         self.intensity = empty_arr
#         self.intensity_sigma = empty_arr
#         self.redundancy = empty_arr
#         self.scale = 1

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
            out = attributes['reduced']
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
            raise ValueError('Invalid unit type. Allowed: 1/nm, 1/A.  Given: {}'.format(val))
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

    @property
    def scale(self):
        return self['entry/processed/scale'][0]

    @scale.setter
    def scale(self,val):
        val_arr = numpy.array([val])
        dts = h5z.DatasetH5(source_dataset=val_arr, name='scale')
        dts.attrs['long_name'] = 'Scale factor applied to the original data. Usually derived from primary beam.'
        self['entry/processed/scale'] = dts

    @property
    def parents(self):
        try:
            return self['entry/data/parents']
        except KeyError:
            logging.debug('File does not have any parents: {}'.format(self.path))
            return None

    @parents.setter
    def parents(self, arr):
        if isinstance(arr, str) or isinstance(arr, h5z.InstrumentFileH5):
            arr = [arr]
        arr_out = []
        for parent in arr:
            if isinstance(parent, str):
                arr_out.append(parent)
            elif isinstance(parent, h5z.InstrumentFileH5):
                arr_out.append(parent.path)
            else:
                raise ValueError('Something unexpected in file name of a parent.') #TODO: Make it some more informative...

        dts =  h5z.DatasetH5(source_dataset=numpy.array(arr_out, dtype='S'), name='parents')
        self['entry/data/parents'] = dts

    def update_attributes(self):
        '''
        Updates file attributes
        '''
        self.attrs['reduced'] = True
        self.attrs['HDF5_Version'] = h5z.hdf5_version
        self.attrs['creator'] = 'AAres {}'.format(aares.version)
        self.attrs['file_time'] = datetime.datetime.now().isoformat()
        self.attrs['base_type'] = base_class.__name__

    def write(self, *args, **kwargs):
        '''
        Writes the file
        '''
        self.update_attributes()
        super(type(self), self).write(*args, **kwargs)

    cls_1D = type("Reduced1D", (base_class,),
                  {
                      "__init__":        __init__,
                      "is_type":         is_type,
                      "q_values":        q_values,
                      "q_value_units":   q_value_units,
                      "intensity":       intensity,
                      "intensity_units": intensity_units,
                      "intensity_sigma": intensity_sigma,
                      "redundancy":      redundancy,
                      "scale":           scale,
                      "parents":         parents,
                      "update_attributes": update_attributes,
                      "write":            write,
                  })
    return cls_1D


def test_Reduced1D_write():
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
    hd1.scale = 45.4678
    hd1.parents=[fin, header]
    hd1.intensity
    hd1.intensity_sigma
    hd1.q_values
    hd1.redundancy
    hd1.q_value_units
    hd1.intensity_units
    hd1.scale
    hd1.parents
    hd1.write('AgBeh_826mm_reduced.h5')

    assert reduced1d.is_type('AgBeh_826mm_reduced.h5')

def test_Reduced1D_read():

    assert Reduced1D_factory.is_type('AgBeh_826mm_reduced.h5')