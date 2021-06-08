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
from abc import ABC, abstractmethod
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


def walk_compress(in_object, out_object, skip=[], __full_path='', log=False, **filters):
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
                walk_compress(in_obj, out_obj, skip=skip, log=log, **filters)
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
            if isinstance(source_dataset,np.ndarray):
                if np.isfortran(source_dataset):
                    order = 'F'
                else:
                    order = 'C'
            else:
                order = 'C'
            obj = super().__new__(self, source_dataset.shape, source_dataset.dtype, buffer=source_dataset[()], order=order)
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

    def write(self, parent, bigger_than = 100, compression = 'gzip', **kwargs):
        '''
        Function used for writing the group to the H5 file
        :param parent: Parent group to be saved in
        :return: True if success
        '''

        if self.size < bigger_than:
            compression = None

        dset = parent.create_dataset(self.name, data=self, compression=compression, **kwargs)
        dset.attrs.update(self.attrs)


def ItemH5(item_in):
    """
    Reads in item from the file, and returns it. It is generic wrapper for reading GroupH5 and DatasetH5
    :param item_in: Item to be converted/read
    :return: GroupH5 or DatasetH5
    """

    if isinstance(item_in, h5py.Group) or isinstance(item_in,GroupH5):
        return GroupH5(item_in)
    elif isinstance(item_in, h5py.Dataset) or isinstance(item_in,DatasetH5):
        return DatasetH5(item_in)
    else:
        raise TypeError('The input type has to be h5py.Group, h5py.Dataset, GroupH5, or DatasetH5. Given: {}'.format(type(item_in)))


class GroupH5(dict):
    """
    Class mimiking h5py.Group

    Instances of this class can be entirely hold in memory without any connection to a file. The instance is also
    picklable, therefore can be used to pass around data during multiprocessing with concurrent.futures.

    The class mimiks h5py.Group, however, the feature set might not be complete and it can evolve with the project.
    """

    def __init__(self, source_group=None, name=None, exclude=None):
        if exclude is None:
            exclude = []

        self.attrs = {}
        self.name = name
        if (isinstance(source_group, h5py.Group) or
                isinstance(source_group, GroupH5) or
                isinstance(source_group, h5py.File)):
            for key, obj in source_group.items():
                if obj.name.strip('/') in exclude:
                    continue
                elif isinstance(obj, h5py.Group):
                    self[key] = GroupH5(obj, exclude=exclude)
                elif isinstance(obj, h5py.Dataset):
                    self[key] = DatasetH5(source_dataset=obj)
                else:
                    raise TypeError('Unknown HDF5 element type: {}'.format(type(obj)))
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

    def __setitem__(self, key, value):
        """
        Enables getting items to nested dictionaries, with syntax like: group5['/foo/bar/foobar']
        """
        split_key = key.strip('/').split('/', maxsplit=1)

        if split_key == ['']:
            raise KeyError
        elif len(split_key) == 1:
            dict.__setitem__(self, split_key[0],value)
        else:
            if split_key[0] not in self:
                dict.__setitem__(self, split_key[0], GroupH5(name=self.name+'/'+split_key[0]))
            if not isinstance(self[split_key[0]], GroupH5):
                raise KeyError('The existing key cannot be extended, because it is not GroupH5 object. "{}"'.format(key))

            self[split_key[0]][split_key[1]] = value

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

    def write(self, parent, **kwargs):
        '''
        Function used for writing the group to the H5 file
        :param parent: Parent group to be saved in
        :return: True if success
        '''

        try:
            me_h5 = parent.create_group(self.name)
        except ValueError:
            me_h5 = parent[self.name]
        me_h5.attrs.update(self.attrs)
        for key, val in self.items():
            val.write(me_h5, **kwargs)

    def walk(self):
        '''
        Returns list of keys of all items and their subitems
        :return:
        '''

        out_items = []
        for key, it in self.items():
            out_items.append(it.name)
            if isinstance(it, GroupH5):
                out_items.extend(it.walk())

        return out_items

class InstrumentFileH5(ABC):
    """
    Abstract class describing generic H5-like files to be used as files containing data
    """
    @property
    @abstractmethod
    def geometry_fields(self):
        '''
        List of items, which defines geometry of the instrument. These items are used for determining, if geometries of two experiments are the same.
        :return: list
        '''
        return []

    @property
    @abstractmethod
    def skip_entries(self):
        '''
        List of items not read from the file at the time of object creation.
        :return: list
        '''
        return []

    @staticmethod
    @abstractmethod
    def is_type(val):
        '''
        Method, which checks, if the item is of the proper type

        :return: bool
        '''
        return True

    def __getitem__(self, key):
        if key in self.skip_entries:
            with FileH5Z(self.abs_path,'r') as h5f:
                val = ItemH5(h5f[key])
        else:
            val = self._h5[key]
        return val

    def __setitem__(self, key, value):
        self._h5[key] = value

    @property
    def _h5(self):
        try:
            val = self.__h5
        except AttributeError:
            val = GroupH5()
            self.__h5 = val
        return val

    @_h5.setter
    def _h5(self,val):
        self.__h5 = val

    @property
    def attrs(self):
        return self._h5.attrs

    def read_header(self, path=None, **filters):
        #TODO: create as constructor option; store in member variable, if one of those attempted to read later, reopen the file

        if path is None:
            try:
                path = self.attrs['abs_path']
            except AttributeError:
                raise AttributeError('Path to the file was not given or set.')
        #TODO : handle missing/wrong file exceptions
        with FileH5Z(path, 'r', **filters) as h5z:
            # walk(h5z, self._h5, skip=skip_entries, **filters)
            # for key, val in h5z.attrs.items():
            #    self.attrs[key] = val
            self._h5 = GroupH5(h5z, exclude=self.skip_entries)

    def write(self, fout, mode='w', skipped=False, **kwargs):
        '''
        Write the object to H5 file

        :param fout:
        :param kwargs:
        :return:
        '''

        with h5py.File(fout, mode=mode) as h5out:
            h5out.attrs.update(self.attrs)
            self._h5.write(h5out, **kwargs)
            if skipped:
                for item in self.skip_entries:
                    try:
                        self[item].write(h5out, **kwargs)
                    except ValueError:
                        pass

    def walk(self):
        '''
        Returns list of items in the object
        :return:
        '''

        return self._h5.walk()


class SaxspointH5(InstrumentFileH5):
    """
    A class holding information on a H5Z file (header snippet). Uses H5-style dictionary

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
        '''

        :param path: Path to the file to be read into the object
        '''
        # self.__temp = tempfile.TemporaryFile()
        # self._h5 = h5py.File(self.__temp, 'a')
        # self._h5

        if isinstance(path, GroupH5):
            self._h5 = copy.deepcopy(path)
        elif isinstance(path, h5py.Group):
            self._h5 = GroupH5(path)
        elif is_h5_file(path):
            if not self.is_type(path):
                raise TypeError('Input is not compatible with SaxspointH5 class.')
            self.read_header(path)

            self.attrs['path'] = path
            self.update_abs_path()
        elif not os.path.isfile(path):
            raise OSError('File not found.')
        else:
            raise TypeError

    @staticmethod
    def is_type(val):
        '''
        Checks for attributes to validate SAXSpoint
        '''
        attributes = {}
        if is_h5_file(val):
            with  FileH5Z(val, 'r') as fin:
                attributes.update(fin.attrs)
        elif isinstance(val, GroupH5) or isinstance(val,h5py.Group):
            attributes.update(val.attrs)
        else:
            raise TypeError('Input should be h5py.Group-like object.')

        try:
            if 'saxsdrive' in attributes['creator'].decode().lower():
                out = True
            else:
                out = False
        except KeyError:
            out = False

        return out

    def update_abs_path(self,abs_path = None):
        '''
        Updates abs_path entry. Based on "path", if intput is None
        :param abs_path:
        :return:
        '''

        if abs_path is not None:
            self.abs_path = abs_path
        else:
            try:
                self.abs_path = os.path.abspath(self.attrs['path'])
            except OSError:
                raise OSError('The data file does not exist: {}'.format(self.path))

    @property
    def sample_name(self):
        return ''.join([chr(i) for i in self['entry/data/sample_name'][0] if i > 0])

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

        detector_offset_y = self['entry/instrument/detector/x_translation'].item()
        detector_offset_x = self['entry/instrument/detector/height'].item()
        return detector_offset_x, detector_offset_y

    @property
    def beam_center_px(self):
        """
        Returns position of primary beam in pixel coordinates
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
        File_time string in isoformat in microseconds
        """
        tm = self.attrs['file_time'].decode()

        return tm[:26] + tm[27:]

    @property
    def data(self):
        """
        Actual data
        :return: numpy.array
        """
        return self['entry/data/data']

    @property
    def path(self):
        """
        File path
        :return:
        """
        return self.attrs['path']
    @path.setter
    def path(self, val):
        self.attrs['path'] = val

    @property
    def abs_path(self):
        """
        File absolute path
        :return:
        """
        return self.attrs['abs_path']
    @abs_path.setter
    def abs_path(self, val):
        self.attrs['abs_path'] = val

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
