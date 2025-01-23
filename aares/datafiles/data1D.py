import datetime
import os.path
import sys
from abc import abstractmethod,ABC

from numpy import dtype

import h5z, h5py
import aares

import numpy
import logging

detector_file_types = {'SaxspointH5': h5z.SaxspointH5,
                   #'Reduced1D': aares.datafiles.data1D.Reduced1D,
                   #'Reduced1D': Reduced1D
                   }

def write_atsas(q_val, avr, std, file_name, header=[], footer=[], separator=" "):
    """
    Writes 1D curve in ATSAS format
    :param q_val: q_values
    :param avr: avreages
    :param std: errors
    :param file_name: output file name
    :param header: lines to be written in the beginging
    :param footer: lines to be written after the data
    :param separator: column separator
    :return:
    """

    assert len(q_val) == len(avr) == len(std)

    try:
        with open(file_name,'w') as fiout:
            fiout.writelines(header)

            for q, a, e in zip(q_val, avr, std):
                line = separator.join(['{}']*3)+os.linesep
                fiout.write(line.format(q,a,e))

            fiout.writelines(footer)
    except PermissionError:
        raise aares.RuntimeErrorUser('Cannot write to {}.'.format(file_name))



class Data1D_meta(ABC):

    @staticmethod
    @abstractmethod
    def is_type(value):
        pass

    @staticmethod
    def base_class_name(fin_path):
        try:
            if h5z.is_h5_file(fin_path):
                with h5z.FileH5Z(fin_path,'r') as fin:
                    return fin.attrs['base_type']
            elif isinstance(fin_path, h5z.GroupH5) or isinstance(fin_path, h5py.Group):
                return fin_path.attrs['base_type']
        except KeyError:
            raise ValueError('This is not a Data1D file.')
        else:
            raise ValueError('This is not a Data1D file.')

    def __init__(self, path, exclude=True):
        '''
        On new instance of the class, one attribute has to be provided:
          * path to a file
          * subclass of h5z.InstrumentFileH5
          * object of class h5z.InstrumentFileH5
        '''
        if isinstance(path, h5z.InstrumentFileH5):
            data_type = Reduced1D_factory(base_class=type(path))
            self._data1d = data_type(path._h5)
        elif isinstance(path, Data1D_meta):
            data_type = Reduced1D_factory(base_class=detector_file_types[path.attrs['base_type']])
            self._data1d = data_type(path, exclude=exclude)
        # elif isinstance(path, type) and issubclass(path, h5z.InstrumentFileH5):
        #     data_type = Reduced1D_factory(base_class=path)
        elif os.path.isfile(path) and self.is_type(path):
            base_type = detector_file_types[self.base_class_name(path)]
            data_type = Reduced1D_factory(base_class=base_type)
            self._data1d = data_type(path, exclude=exclude)
        else:
            raise ValueError('Unexpected type of parameter was recieved: {}'.format(type(path)))

    def __getattr__(self, item):
        # Check if '_data1d' is in the instance dictionary
        if '_data1d' in self.__dict__ and hasattr(self._data1d, item):
            return getattr(self._data1d, item)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        # If the attribute is '_data1d' or not yet initialized, directly set it
        if '_data1d' in self.__dict__ and key in dir(self._data1d): #hasattr(self._data1d, key):
            setattr(self._data1d, key, value)
        elif key == '_data1d' or key not in self.__dict__:
            super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def __getitem__(self,item):
        return self._data1d[item]

    def __setitem__(self, key, value):
        self._data1d[key] = value

class Reduced1D(Data1D_meta):
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




class Subtract1D(Data1D_meta):
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
            out = attributes['subtracted']
        except KeyError:
            out = False

        return out

    def update_attributes(self):
        '''
        Updates file attributes
        '''
        #self.attrs['reduced'] = True
        self.attrs['subtracted'] = True
        self.attrs['HDF5_Version'] = h5z.hdf5_version
        self.attrs['creator'] = 'AAres {}'.format(aares.version)
        #self.attrs['file_time'] = datetime.datetime.now().isoformat()
        #self.attrs['base_type'] = base_class.__name__


def Reduced1D_factory(base_class=(h5z.SaxspointH5,)):
    _skip_entries = [
        'entry/data/Idev',
        'entry/data/I',
        'entry/data/redundancy',
        'entry/data/Iscale',
        'entry/data/Q',
    ]
    assert issubclass(base_class, h5z.InstrumentFileH5)
    def __init__(self, path, exclude=True):

        if isinstance(path, Data1D_meta):
            super(type(self), self).__init__(path._data1d) #, skip_entries=_skip_entries)
            if not exclude:
                for entry in self.skip_entries: #_skip_entries:
                    try:
                        self[entry] = path[entry]
                    except KeyError:
                        logging.debug('Skipping entry "{}" because it does not exist.'.format(entry))
        else:
            super(type(self), self).__init__(path) #, skip_entries=_skip_entries)

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
    def skip_entries(self):
        return _skip_entries

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
        dts.attrs['uncertainties']='Idev'
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
            return [it.decode() for it in self['entry/data/parents'].tolist()]
            #return self['entry/data/parents']
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

        self['entry/process'] = proc

    def write(self, *args, **kwargs):
        '''
        Writes the file
        '''
        self.update_attributes()
        super(type(self), self).write(*args, **kwargs)

    def export(self, output, params=None):

        if params is None:
            params = aares.common.phil_export.extract().export
        else:
            aares.common.phil_export.format(params)

        if self.q_value_units not in ['1/nm', '1/angstrom', '1/m']:
            if len(self.q_values[self.q_values < 1]) > len(self.q_values):
                in_units = '1/angstrom'
            else:
                in_units = '1/nm'
            logging.info('Units of Q-value guessed as: {}'.format(in_units))
        else:
            in_units = self.q_value_units

        if in_units == '1/angstrom':
            in_q_multipiler = 10
        elif in_units == '1/m':
            in_q_multipiler = 1e-9
        elif in_units == '1/nm':
            in_q_multipiler = 1
        else:
            raise ValueError('Invalid input Q-units: {}'.format(in_units))

        if params.units.q == '1/nm':
            out_q_multipiler = 1
        elif params.units.q == '1/angstrom' or params.units.q == '1/A':
            out_q_multipiler = 0.1
        else:
            raise ValueError('Invalid output units: {}'.format(params.units.q))

        q_units_multipiler = in_q_multipiler * out_q_multipiler

        parents = self.parents if self.parents is not None else []

        header =[f'Sample description: {self.sample_name}\n',
                 f'Sample: c= 0 mg/ml Code:\n',
                 'Parent(s): ' + ' '.join(parents),
                 '\n']

        footer = ['range-from: 1\n',
                  f'range-to: {len(self.q_values)}\n',
                  f'creator: {sys.argv[0]}\n',
                  f'creator-version: {aares.__version__}',]
        footer.extend('\nparent: '.join(['']+parents))

        write_atsas(self.q_values*q_units_multipiler, self.intensity, self.intensity_sigma,
                                 file_name=output,
                                 header=header,
                                 footer=footer,)



    cls_1D = type("Data1D", (base_class,),
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
                      "skip_entries":    skip_entries,
                      "export":          export,
                  })
    return cls_1D


