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


class DatasetH5(np.ndarray):
    """
    Class mimiking h5py.Dataset

    Holds the data in a Numpy array, and adds attributes dictionary DatasetH5.attrs
    """

    def __new__(self, source_dataset=None, name=None, *args, **kwargs):

        if source_dataset is not None:
            return super().__new__(self, source_dataset.shape, source_dataset.dtype, buffer=source_dataset[()])
        else:
            return super().__new__(self, *args, **kwargs)

    def __init__(self, source_dataset=None, name=None, *args, **kwargs):
        self.attrs = {}
        if name is not None:
            self.name = name

        if isinstance(source_dataset, h5py.Dataset):
            for key, val in source_dataset.attrs.items():
                self.attrs[key] = copy.copy(val)
            self.name = source_dataset.name


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


class SaxspointH5():
    """
    A class holding informations on a H5Z file (header snippet). Uses H5-style dictionary

    Note, that this might be extended, as projects progresses
    """

    def __init__(self, path):
        # self.__temp = tempfile.TemporaryFile()
        # self.__h5 = h5py.File(self.__temp, 'a')
        # self.__h5

        self.read_header(path)

        self.attrs['path'] = path
        self.attrs['abs_path'] = os.path.abspath(self.attrs['path'])

    def __getitem__(self, key):
        return self.__h5[key]

    def __setitem__(self, key, value):
        self.__h5[key] = value

    @property
    def attrs(self):
        return self.__h5.attrs

    def read_header(self, path=None, **filters):
        skip_entries = ['entry/data/data',
                        'entry/data/x_pixel_offset',
                        'entry/data/y_pixel_offset',
                        'entry/instrument/detector/data',
                        #'entry/instrument/detector/x_pixel_offset',
                        #'entry/instrument/detector/y_pixel_offset',
                        ]
        if path is None:
            try:
                path = self.attrs['abs_path']
            except AttributeError:
                raise AttributeError('Path to the file was not given or set.')
        with FileH5Z(path, 'r', **filters) as h5z:
            # walk(h5z, self.__h5, skip=skip_entries, **filters)
            # for key, val in h5z.attrs.items():
            #    self.attrs[key] = val
            self.__h5 = GroupH5(h5z, exclude=skip_entries)

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


def test_DatasetH5():
    a = np.full(1, np.nan)
    dts = DatasetH5(a)
    print(dts)


if __name__ == "__main__":
    test_DatasetH5()
