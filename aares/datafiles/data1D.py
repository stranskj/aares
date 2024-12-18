import datetime
import os.path

from numpy import dtype

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
        return self['entry/data/Q']

    @q_values.setter
    def q_values(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='Q')
        dts.attrs['long_name'] = 'Coordinate in reciprocal space'
        dts.attrs['units'] = 'Unknown'
        dts.attrs['resolution_description'] = 'Bin'
        self['entry/data/Q'] = dts

    @property
    def q_value_units(self):
        return self['entry/data/Q'].attrs['units']

    @q_value_units.setter
    def q_value_units(self, val):
        if val not in ['1/nm', '1/angstrom', '1/m']:
            raise ValueError('Invalid unit type. Allowed: 1/nm, 1/angstrom, 1/m.  Given: {}'.format(val))
        self['entry/data/Q'].attrs['units'] = val

    @property
    def intensity(self):
        return self['entry/data/I']

    @intensity.setter
    def intensity(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='I')
        dts.attrs['long_name'] = 'Scattering intensity'
        dts.attrs['units'] = 'arbitrary'
        dts.attrs['uncertainities']='Idev'
        self['entry/data/I'] = dts


    @property
    def intensity_units(self):
        return self['entry/data/I'].attrs['units']

    @intensity_units.setter
    def intensity_units(self, val):
        if val not in ['1/m',  '1/cm', 'arbitrary']:
            raise ValueError('Invalid unit type. Allowd: 1/m, 1/cm, arbitrary.  Given: {}'.format(val))
        self['entry/data/I'].attrs['units'] = val
        try:
            self['entry/data/Idev'].attrs['units'] = val
        except KeyError:
            logging.debug('Idev entry in the file does not exist yet.')

    @property
    def intensity_sigma(self):
        return self['entry/data/Idev']

    @intensity_sigma.setter
    def intensity_sigma(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='Idev')
        dts.attrs['long_name'] = 'Sigma error of scattering intensity'
        dts.attrs['units'] = self.intensity_units
        self['entry/data/Idev'] = dts

    @property
    def redundancy(self):
        return self['entry/data/redundancy']

    @redundancy.setter
    def redundancy(self, val):
        dts = h5z.DatasetH5(source_dataset=val, name='redundancy')
        dts.attrs['long_name'] = 'Number of pixels in q-bin'
        dts.attrs['units'] = 'pixels'
        self['entry/data/redundancy'] = dts

    @property
    def scale(self):
        return self['entry/data/Iscale'][0]

    @scale.setter
    def scale(self,val):
        val_arr = numpy.array([val])
        dts = h5z.DatasetH5(source_dataset=val_arr, name='Iscale')
        dts.attrs['long_name'] = 'Scale factor applied to the original data. Usually derived from primary beam.'
        self['entry/data/Iscale'] = dts

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
        #self.attrs['file_time'] = datetime.datetime.now().isoformat()
        self.attrs['base_type'] = base_class.__name__

    def add_process(self, name=None, description=None):
        '''
        Adds NXcanSAS compatible NXprocess
        '''

        proc = h5z.GroupH5(name='process')
        proc.attrs['canSAS_class'] = 'SASprocess'
        proc.attrs['NX_class'] = 'NXprocess'
        proc['date'] = h5z.DatasetH5(source_dataset=numpy.array(datetime.datetime.now().isoformat(), dtype='S'), name='date')
        if name is not None:
            proc['name'] = h5z.DatasetH5(source_dataset=numpy.array(name, dtype='S'), name='name')
        if description is not None:
            proc['description'] = h5z.DatasetH5(source_dataset=numpy.array(description, dtype='S'), name='description')

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
                      "write":           write,
                      "add_process":     add_process,
                  })
    return cls_1D


