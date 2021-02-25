"""
Handling of h5z files


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

from contextlib import contextmanager
from zipfile import ZipFile
import zipfile
import h5py
import os
import logging
#from saxspoint import my_print
import copy
import numpy as np


@contextmanager
def FileH5Z(name, file_mode='r', *args, **kwargs):
    """
    args, kwargs are passed to h5py
    """
    try:
        if file_mode == 'r':
            try:
                with ZipFile(name, file_mode) as zip_in:
                    zip_mem = zip_in.namelist()[0]
                    with zip_in.open(zip_mem, 'r') as mem:
                        with h5py.File(mem, file_mode, *args, **kwargs) as h5f:
                            yield h5f
            except zipfile.BadZipFile:
                with h5py.File(name, file_mode, *args, **kwargs) as h5f:
                    yield h5f
        else:
            raise NotImplementedError('file_mode = {}'.format(file_mode))
    finally:
        pass

def is_h5_file(name):
    '''
    Chekcs, if the file is readable h5 or h5z readable file
    :param name: File name
    :type name: str
    :rtype: bool
    '''

    try:
        with FileH5Z(name, 'r') as h5f:
            return True
    except OSError:
        return False


def _report(operation, key, obj):
    type_str = type(obj).__name__.split(".")[-1].lower()
    print("%s %s: %s." % (operation, type_str, key))


def h5py_compatible_attributes(in_object):
    """Are all attributes of an object readable in h5py?"""
    try:
        # Force obtaining the attributes so that error may appear
        [0 for at in in_object.attrs.items()]
        return True
    except:
        return False


def copy_attributes(in_object, out_object):
    """Copy attributes between 2 HDF5 objects."""
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value


def walk(in_object, out_object, skip=[], __full_path='', log=False, **filters):
    """Recursively copy&compress the tree.

    If attributes cannot be transferred, a copy is created.
    Otherwise, dataset are compressed.
    """

    for key, in_obj in in_object.items():
        if in_obj.name.strip('/') in skip:
            continue
        elif not isinstance(in_obj, h5py.Datatype) and h5py_compatible_attributes(in_obj):
            if isinstance(in_obj, h5py.Group):
                out_obj = out_object.create_group(key)
                walk(in_obj, out_obj, skip=skip, log=log, **filters)
                if log:
                    _report("Copied", key, in_obj)
            elif isinstance(in_obj, h5py.Dataset):
                try:
                    if len(in_obj) > 3:
                        out_obj = out_object.create_dataset(key, data=in_obj, **filters)
                    else:
                        out_obj = out_object.create_dataset(key, data=in_obj)
                except TypeError:
                    out_obj = out_object.create_dataset(key, data=in_obj)
                if log:
                    _report("Compressed", key, in_obj)
            else:
                raise "Invalid object type %s" % type(in_obj)
            copy_attributes(in_obj, out_obj)
        else:
            # We copy datatypes and objects with non-understandable attributes
            # identically.
            if log:
                _report("Copied", key, in_obj)
            in_object.copy(key, out_object)


def recompress(path1, path2, log=False, compression='gzip'):
    """Compress a HDF5 file.

    :param path1: Input path
    :param path2: Output path
    :param log: Whether to print results of operations'
    :returns: A tuple(original_size, new_size)
    """
    with h5py.File(path1, "r") as in_file, h5py.File(path2, "w") as out_file:
        walk(in_file, out_file, log=log, compression=compression)
    return os.stat(path1).st_size, os.stat(path2).st_size

def groupH5_to_scope(group):
    '''

    :param group:
    :return:
    '''
    pass

def datasetH5_to_scope(dataset, max_length=20):
    '''
    Converts DatasetH5 or h5py.dataset to PHIL scope.

    Conversion:
    name is name
    attributes -> subscope called "attributes"
    actual value -> subdefinition called "value". If the array is longer than `max_length`, only shape of the array is stored. If `None`, store everything.
    attribute `long_name` -> .help
    value type -> translates to value/.type

    :param dataset:
    :return:
    '''


def h5_to_phil(header):
    '''
    Converts H5-like structure to PHIL scope
    :param header:
    :rtype: phil.scope
    '''


def phil_to_h5(scope):
    '''
    Converts PHIL scope to H5-like structure
    :param scope: Input data
    :type scope:  phil.scope
    :rtype: GroupH5
    '''
    pass

class DatasetH5(np.ndarray):
    """
    Class mimicking h5py.Dataset

    Holds the data in a Numpy array, and adds attributes dictionary DatasetH5.attrs

    Getting data using `[]` returns vanilla numpy.array
    """

    def __new__(self, source_dataset=None, name=None, *args, **kwargs):

        if source_dataset is not None:
            obj = super().__new__(self, source_dataset.shape, source_dataset.dtype, buffer=source_dataset[()])
        else:
            obj = super().__new__(self, *args, **kwargs)
        obj.attrs = {}
        obj.name = name
        return obj

    def __init__(self, source_dataset=None, name=None, *args, **kwargs):

        if name is not None:
            self.name = name

        if isinstance(source_dataset, h5py.Dataset):
            for key, val in source_dataset.attrs.items():
                self.attrs[key] = copy.copy(val)
            self.name = source_dataset.name


    # It has to be here because standard np.array childs have problems with pickling new attributes
    # It is important for multiprocessing to work
    # self.attrs = getattr(obj, 'attrs', None)
    def __array_finalize__(self, obj):
        if obj is None: return
        self.attrs = getattr(obj, 'attrs', None)
        self.name = getattr(obj, 'name', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(DatasetH5, self).__reduce__()
        # Create our own tuple to pass to __setstate__, but append the __dict__ rather than individual members.
        new_state = pickled_state[2] + (self.__dict__,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])  # Update the internal dict from state
        # Call the parent's __setstate__ with the other tuple elements.
        super(DatasetH5, self).__setstate__(state[0:-1])

    # End of the pickling stuff

    def __eq__(self, other): # It breaks numpy's elements vice comparison
        if not (isinstance(other, DatasetH5) or isinstance(other,h5py.Dataset)):
            # don't attempt to compare against unrelated types
            return NotImplemented
        attr_bool = True
        try:
            attr_bool = len(self.attrs) == len(other.attrs)
            for key, val in self.attrs.items():
                attr_bool = attr_bool and (val==other.attrs[key])
        except KeyError:
            attr_bool = False
        except AttributeError:
            pass
        return  attr_bool and np.array_equal(self, other)

    # Custom might fix previous...
    def __getitem__(self, item):
        return np.array(super().__getitem__(item))

class GroupH5(dict):
    """
    Class mimiking h5py.Group

    Instances of this class can be entirely hold in memory without any connection to a file. The instance is also
    picklable, therefore can be used to pass around data during multiprocessing with concurrent.futures.

    The class mimiks h5py.Group, however, the feature set might not be complete and it can evolve with the project.
    """

    def __init__(self, source_group=None, exclude=[]):
        self.attrs = {}
        if (isinstance(source_group, h5py.Group) or
                isinstance(source_group, GroupH5) or
                isinstance(source_group, h5py.File)):
            for key, obj in source_group.items():
                if obj.name.strip('/') in exclude:
                    continue
                elif isinstance(obj, h5py.Group):
                    self[key] = GroupH5(obj, exclude)
                elif isinstance(obj, h5py.Dataset):
                    self[key] = DatasetH5(source_dataset=obj)
                else:
                    raise TypeError('Unknown HDF5 element type: {}'.format())
            for key, val in source_group.attrs.items():
                self.attrs[key] = copy.copy(val)
            # self.__dict__.update(source_group.__dict__)
            self.name = source_group.name

    def __getitem__(self, key):
        """
        Enables getting items from nested dictionaries, with syntax like: group5['/foo/bar/foobar']
        """
        split_key = key.strip('/').split('/', maxsplit=1)
        try:
            if split_key == ['']:
                return self
            elif len(split_key) == 1:
                return dict.__getitem__(self, split_key[0])
            else:
                return self[split_key[0]][split_key[1]]
        except KeyError:
            raise KeyError('Entry does not exsist: {}'.format(key))

    def __eq__(self, other):
        if not (isinstance(other, GroupH5) or isinstance(other,h5py.Group)):
            # don't attempt to compare against unrelated types
            return NotImplemented
        try:
            attr_bool = len(self.attrs) == len(other.attrs)
            for key, val in self.attrs.items():
                attr_bool = attr_bool and (val==other.attrs[key])
        except KeyError or AttributeError:
            attr_bool = False

        return  super().__eq__(other) and attr_bool



class SaxspointH5():
    """
    A class holding informations on a H5Z file (header snippet). Uses H5-style dictionary

    Note, that this might be extended, as projects progresses

    :ivar geometry_fields: List of fields, which describe the experiment geometry
    """
    skip_entries = ['entry/data/data',
                        'entry/data/x_pixel_offset',
                        'entry/data/y_pixel_offset',
                        'entry/instrument/detector/data',
                        #'entry/instrument/detector/x_pixel_offset',
                        #'entry/instrument/detector/y_pixel_offset',
                    ]

    geometry_fields = [
        'entry/instrument/detector/depends_on',
        'entry/instrument/detector/description',
        'entry/instrument/detector/detector_number',
        'entry/instrument/detector/distance',
        'entry/instrument/detector/height',
        'entry/instrument/detector/meridional_angle',
        'entry/instrument/detector/sensor_material',
        'entry/instrument/detector/sensor_thickness',
        'entry/instrument/detector/x_pixel_offset',
        'entry/instrument/detector/x_pixel_size',
        'entry/instrument/detector/y_pixel_offset',
        'entry/instrument/detector/y_pixel_size',
        'entry/instrument/detector/x_translation',
        'entry/instrument/monochromator',
    ]

    def __init__(self, path):
        # self.__temp = tempfile.TemporaryFile()
        # self.__h5 = h5py.File(self.__temp, 'a')
        # self.__h5

        self.read_header(path)

        self.attrs['path'] = path
        self.attrs['abs_path'] = os.path.abspath(self.attrs['path'])

    def __getitem__(self, key):
        if key in self.skip_entries:
            with FileH5Z(self.abs_path,'r') as h5f:
                val = h5f[key]
        else:
            val = self.__h5[key]
        return val

    def __setitem__(self, key, value):
        self.__h5[key] = value

    @property
    def attrs(self):
        return self.__h5.attrs

    def read_header(self, path=None, **filters):
        #TODO: create as constructor option; store in member variable, if one of those attempted to read later, reopen the file

        if path is None:
            try:
                path = self.attrs['abs_path']
            except AttributeError:
                raise AttributeError('Path to the file was not given or set.')
        #TODO : handle missing/wrong file exceptions
        with FileH5Z(path, 'r', **filters) as h5z:
            # walk(h5z, self.__h5, skip=skip_entries, **filters)
            # for key, val in h5z.attrs.items():
            #    self.attrs[key] = val
            self.__h5 = GroupH5(h5z, exclude=self.skip_entries)

    @property
    def pixel_size(self):
        """
        Pixel size of the detector in meters
        """
        pixel_size_x = self['entry/instrument/detector/x_pixel_size'].item()
        pixel_size_y = self['entry/instrument/detector/y_pixel_size'].item()
        return pixel_size_x, pixel_size_y

    @property
    def detector_offset(self):
        """
        Returns x,y offset (in meters), so after applying the primary beam is at 0,0
        """
        detector_offset_x = self['entry/instrument/detector/x_translation'].item()
        detector_offset_y = self['entry/instrument/detector/height'].item()
        return detector_offset_x, detector_offset_y

    @property
    def beam_center_px(self):
        """
        Returns postion of primary beam in pixel coordinates
        """
        det = self.detector_offset
        pix = self.pixel_size

        x = -det[0] / pix[0]
        y = -det[1] / pix[1]

        return x, y
    @property
    def frame_size(self):
        '''
        Shape of the frame
        '''
        return self['entry/instrument/detector/pixel_mask'].shape

    @property
    def sdd(self):
        """
        Sample to detector distance
        """
        return self['entry/instrument/detector/distance'].item()

    @property
    def wavelength(self):
        """
        X-ray wavelength
        """
        return self['entry/instrument/monochromator/wavelength'].item()

    @property
    def time_offset(self):
        """
        Time offsets of the frames from file creation
        """
        return self['entry/instrument/detector/time']

    @property
    def file_time_iso(self):
        """
        File_time string in isoformat in micorseconds
        """
        tm = self.attrs['file_time'].decode()

        return tm[:26] + tm[27:]

    @property
    def path(self):
        """
        File path
        :return:
        """
        return self.attrs['path']
    @property
    def abs_path(self):
        """
        File absolute path
        :return:
        """
        return self.attrs['abs_path']


def test_equal_Datasets():
    f1 = 'data/10x1s.h5'
    header = SaxspointH5(f1)
    dataset1 = header['entry/data/sdd']

    f2 = 'data/10x1s.h5'
    header = SaxspointH5(f2)
    dataset2 = header['entry/data/sdd']
    assert dataset1 == dataset2

def test_equal_Groups():
    f1 = 'data/10x60s_363mm_010Frames.h5'
    header = SaxspointH5(f1)
    header1 = header['entry/instrument/detector']

    f2 = 'data/AgBeh_363mm.h5z'
    header = SaxspointH5(f2)
    header2 = header['entry/instrument/detector']

  #  with FileH5Z(f1,'r') as h5z1, FileH5Z(f2,'r') as h5z2:

   #     dc1 = h5z1['entry/sample']
    #    dc2 = h5z2['entry/sample']

     #   assert header2 == dc1
    #header1 = SaxspointH5('data/10x1s.h5')
    #header2 = SaxspointH5('data/10x1s.h5')

    assert header1 == header2

def test_DatasetH5():
    a = np.full(1, np.nan)
    dts = DatasetH5(a)
    print(dts)


if __name__ == "__main__":
    test_equal_Datasets()
    test_equal_Groups()
