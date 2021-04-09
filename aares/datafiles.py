import glob
import itertools
import math
import os
import logging

import freephil as phil

import aares
import h5z
from aares import power as pwr
from aares.import_file import phil_core

phil_files = phil.parse('''
headers = None
.type = path
.help = File storing parsed headers. If the file is not present, headers are read from the original data files.

group 
.multiple = True
.help = Group of files with the same geometry header
.expert_level = 0
{
    name = None
    .expert_level = 0
    .help = Name of the group
    .type = str

    geometry = None
    .type = path
    .help = File with geometry description. Special PHIL file, or H5 with correct header
    .expert_level = 3

    mask = None
    .type = path
    .help = File with the mask to be used.
    .expert_level = 0

    file 
    .multiple = True
    .help = A file in the group
    {
        path = None
        .type = path
        .help = Location of the file. This is required parameter of the "file"

        name = None
        .type = str
        .help = String which is used as a reference for the file elsewhere. It has to be unique string.
    }

}
''')


def longest_common(list_strings, sep='_'):
    """
    Splits strings by `sep`, and return the common sections

    :param str1:
    :param str2:
    :return:
    """

    strips = [string.split(sep) for string in list_strings]

    unique = {key: '' for key in itertools.chain.from_iterable(strips)}

    common = []

    for key in unique.keys():
        if all([key in strp for strp in strips]):
            common.append(key)
    return common


def group_object(files=[], name=None):
    '''
    Returns empty Group scope_extract
    :return:
    '''

    pe = phil_files.extract()
    gr = pe.group.pop()
    gr.file.pop()

    gr.file.extend(files)
    gr.name = name

    return gr


def file_object(path=None, name=None):
    '''
    Returns empty File scope_extract
    :return:
    '''

    pe = phil_files.extract()
    fo = pe.group[0].file.pop()
    fo.path = path
    fo.name = name
    return fo


def search_files(indir, suffix):
    fiout = []
    for root, dirs, files in os.walk(indir):
        for file in files:
            if file.endswith(suffix):
                fiout.append(os.path.join(root, file))
    return fiout


def get_files(inpaths, suffixes):
    if isinstance(suffixes, str):
        suffixes = [suffixes]

    for i, suf in enumerate(suffixes):
        if not (suf[0] == '.'):
            suffixes[i] = '.' + suf

    files = [[file for file in glob.glob(fi)] for fi in inpaths]
    inpaths = [fi for fi in itertools.chain.from_iterable(files)]
    file_list = []
    for fi in inpaths:
        if os.path.isdir(fi):
            for suf in suffixes:
                file_list.extend(search_files(fi, suf))
        else:
            file_list.append(fi)
    return file_list


def files_to_groups(files, headers_to_match='entry/instrument/detector', ignore_merged=True):
    """
    Split the files in the groups based on common headers.

    :param files: list of files to be distributed
    :param headers_to_match: List of headers, which has to match, for files to consider similar
    :param ignore_merged: If True, merged files will be ignored
    :return:
    """
    groups = []
    if isinstance(headers_to_match, str):
        headers_to_match = [headers_to_match]

    if not isinstance(files, dict):
        files = pwr.get_headers_dict(files)

    group_id = 1

    for fi, hd in files.items():
        if is_merged(hd) and ignore_merged:
            continue

        file_scope = file_object(path=fi)

        for group in groups:
            if all(hd[match] == files[group.file[0].path][match] for match in headers_to_match):
                group.file.append(file_scope)
                break
        else:
            groups.append(group_object([file_scope], name='group{:03d}'.format(group_id)))
            group_id += 1

    return groups


def is_merged(file_in):
    '''
    Checkes, whether the file is average of multiple frames
    :param file_in: header or file name
    :type  file_in: h5z.SaxspointH5 or string
    :return:
    '''

    if isinstance(file_in, str):
        try:
            file_in = h5z.SaxspointH5(file_in)
        except:
            raise IOError('Wrong or missing file: {}'.format(file_in))

    try:
        avr = file_in['entry/data/averaged_frames']
        return True
    except KeyError:
        return False


def is_fls(fi):
    '''
    Checks if the file is AAres PHIL file
    :param fi:
    :return:
    '''

    if fi.endswith('.fls'):
        return True
    else:
        return False

class FileHeadersDictionary(dict):
    '''
    A dictionary for storing pre-read file headers (SaxspointH5 objects)
    '''

    def __getitem__(self, key):
        '''
        If the file is not in the dictionary, but exists, it is read to the dictionary
        :param item:
        :return:
        '''

        try:
            item = super().__getitem__(key)
        except KeyError:
            if os.path.exists(key):
                item = h5z.SaxspointH5(key)
                self[key] = item
            else:
                raise KeyError('Entry does not exist: {}'.format(key))
        return item


class DataFilesCarrier:
    """
    Handling class to control file imports

    :ivar files_dict: Dictionary of read file headers. File path as a key.
    :ivar file_groups: Files ordered into the groups (phil.scope_extract)
    :ivar file_scope: Full phil.scope_extract of phil_files
    """


    def __init__(self, run_phil=None, file_phil=None, nproc=None):
        """
        :param run_phil:
        :type  run_phil: phil.scope_extract
        :return: Returns a list of files separated to groups in format of `import_file.phil_file`
        :rtype: phil.scope_extract
        """

        assert (run_phil is not None) ^ (file_phil is not None)

        self._files_dict = FileHeadersDictionary()
        self.file_scope = phil_files.extract()
        self.file_groups = None
        self.nproc = nproc

        if run_phil is not None:
            self.from_input_phil(run_phil)
        elif file_phil is not None:
            self.from_phil_file(file_phil)
        else:
            raise AssertionError('No argument given.')


    def __len__(self):
        return len(self.files_dict)

    @property
    def files_dict(self):
        return self._files_dict

    @files_dict.setter
    def files_dict(self,item):
        self._files_dict = FileHeadersDictionary()
        self._files_dict.update(item)

    @property
    def file_groups(self):
        return self.file_scope.group

    @file_groups.setter
    def file_groups(self, item):
        self.file_scope.group = item

    @property
    def header_file(self):
        return self.file_scope.headers

    @header_file.setter
    def header_file(self, item):
        self.file_scope.headers = item

    def from_input_phil(self, phil_in):
        """
        Processes files using an input parameters in PHIL
        :return:
        """
        try:
            phil_core.format(phil_in)
        except : #TODO: chceck what exception can actually occure
            raise AttributeError('Wrong input Phil parameters.')

        files = get_files(phil_in.search_string, phil_in.suffix)
        aares.my_print('Found {} files. Reading headers...'.format(len(files)))

        if len(files) == 0:
            raise aares.RuntimeErrorUser('No files found.')

        self.files_dict = pwr.get_headers_dict(files, nproc=self.nproc)
        if phil_in.ignore_merged:
            i = 0
            for fi in list(self.files_dict.keys()):
                if is_merged(self.files_dict[fi]):
                    self.files_dict.pop(fi)
                    i += 1
            aares.my_print('Excluded {} merged files.'.format(i))
        groups = files_to_groups(self.files_dict, headers_to_match= h5z.SaxspointH5.geometry_fields ,ignore_merged=phil_in.ignore_merged)
        aares.my_print(
            'Files assigned to {} group(s) by common experiment geometry.'.format(len(groups)))
        self.file_groups = phil_files.extract()
        self.file_groups = groups

        if phil_in.headers is None:
            phil_in.headers = os.path.splitext(phil_in.output)[0]+'.hdr'

        self.header_file = phil_in.headers

        self.set_group_geometries()
        if phil_in.prefix == 'time':
            self.sort_by_time()
            prefix = True
        elif phil_in.prefix == 'file_name':
            prefix = True
        else:
            prefix = False
        self.set_name_from_filename(strip_common=phil_in.shorten, prefix=prefix)

    def from_phil_file(self, phil_in):
        """
        Imports the description of groups from the PHIL file
        :param phil_in: Input PHIL parameters
        :type phil_in: string, phil.scope or phil.scope_exract
        :return:
        """

        if isinstance(phil_in, str):
            if os.path.isfile(phil_in):
                phil_in = phil.parse(file_name=phil_in)
            else:
                raise aares.RuntimeErrorUser('File not found: {}'.format(phil_in))

        if isinstance(phil_in, phil.scope):
            phil_in = phil_files.fetch(phil_in).extract()

        assert isinstance(phil_in, phil.scope_extract)

        self.file_scope = phil_files.format(phil_in).extract()
        if self.header_file is not None:
            if os.path.isfile(self.header_file):
                self.read_headers_from_file()
            else:
                self.read_headers()
                self.write_headers_to_file()
        else:
            self.read_headers()


    def _is_file_key(self, key):
        """
        Checks, if the key is available in the group.file scope
        :param key:
        :rtype: bool
        """
        empty_file = file_object()
        if key not in empty_file.__dict__.keys():
            return False
        else:
            return True

    def set_group_geometries(self):
        """
        Sets the `group.geometry` to the first file of the group

        :return:
        """
        for group in self.file_groups:
            group.geometry = group.file[0].path

    def set_name_from_filename(self, strip_common=True, prefix = True, sep='_'):
        """
        Set group.file.name based on the file name

        :param strip_common: Uses only unique part of the name
        :return:
        """
        for group in self.file_groups:
            for fi in group.file:
                fi.name = fi.path

        if strip_common:
            for group in self.file_groups:
                #Strip file path
                if len(group.file) == 1:
                    group.file[0].name = os.path.splitext(os.path.split(group.file[0].path)[1])[0]
                else:
                    filepaths = [os.path.splitext(fi.path)[0] for fi in group.file]
                    common_path = os.path.commonpath(filepaths)
                    for fi in group.file:
                        if len(common_path) > 0:
                            fi.name = fi.path.split(common_path)[1].strip('.\/_'+sep)
                        else:
                            fi.name = fi.path
                        fi.name = os.path.splitext(fi.name)[0].replace('/','_')

                    common = longest_common([fi.name for fi in group.file])
                    for fi in group.file:
                        name = fi.name.split(sep)
                        for cmn in common:
                            name.remove(cmn)
                        fi.name = sep.join(name)

        if prefix:
            i = 1
            digit = int(math.log10(len(self.files_dict))) + 1
            for fi in self.files_dict.keys():
                scope = self.get_file_scope(fi, key='path')
                scope.name = sep.join([str(i).zfill(digit), scope.name])
                i += 1

    def get_file_scope(self, value, key='name'):
        """
        Searches self.file_groups for a file, whose key match value. Returns respective file scope.

        :param value: Description of the file to be searched for.
        :param key: A key name to be searched in. Has to be one of group.file keys
        :rtype: phil.scope_extract
        """

        if not self._is_file_key(key):
            raise AttributeError('This key type is not used: {}'.format(key))

        for gr in self.file_groups:
            for fi in gr.file:
                if fi.__dict__[key] == value:
                    return fi
        else:
            raise KeyError

    def get_header(self, value, key='name'):
        """
        Returns header of the file, whose key match value.
        :param value: Description of the file to be searched for.
        :param key: A key name to be searched in. Has to be one of group.file keys
        :rtype: SaxspointH5
        """

        file_scope = self.get_file_scope(value, key)
        return self.files_dict[file_scope.path]

    def files(self, key):
        """
        Iterates over file keys
        :param key: Key to iterate over
        :return: keys
        """
        if not self._is_file_key(key):
            raise AttributeError('This key type is not used: {}'.format(key))

        for gr in self.file_groups:
            for fi in gr.file:
                yield fi.__dict__[key]

    def read_headers(self):
        """
        Reads the file headers, and fills the self.files_dict
        """
        self.files_dict = pwr.get_headers_dict(list(self.files('path')), nproc=self.nproc)

    def sort_by_time(self):
        """
        Sort the file dictionary by header time
        """
        sorted_dict = {k: self.files_dict[k] for k in sorted(self.files_dict,
                                                             key=lambda x:self.files_dict[x].file_time_iso)}

        self.files_dict = sorted_dict


    def write_groups(self,file_out='files.fls'):
        '''
        Write the file groups to a file
        :return:
        '''

        try:
            group_out = self.file_scope
            with open(file_out, 'w') as fiout:
                phil_files.format(group_out).show(out=fiout)
        except PermissionError:
            aares.RuntimeErrorUser('Cannot write to {}. Permission denied.'.format(fiout))

    def write_headers_to_file(self,file_out=None):
        '''
        Writes the object to a file
        :param file_out:
        :return:
        '''
        import h5py

        if file_out is None:
            file_out = self.header_file
        assert file_out is not None

        master_group = h5z.GroupH5()
        master_group['file_headers'] = h5z.GroupH5()

        i = 0
        digit = int(math.log10(len(self.files_dict))) + 1
        for name, header in self.files_dict.items():
            key =str(i).zfill(digit)
            master_group['file_headers/'+key] = header._h5
            i += 1

        master_group.attrs['aares_file_type'] = 'data_file_headers'
  #      master_group.attrs['aares_version'] = aares.version
    # TODO: verze se nevraci jako string
    # FAILING: jednotlive sub Groupy maji spatne jmeno, protoze nejsou zalozene na tomto...
        try:
            with h5py.File(file_out, 'w') as fiout:
                master_group.write(fiout, compression='gzip')
        except PermissionError:
            aares.RuntimeErrorUser('Cannot write to {}. Permission denied.'.format(file_out))

    def read_headers_from_file(self,file_in=None):
        '''
        Reads the serialized headers from a file
        :param file_in:
        :return:
        '''
        import h5py

        if file_in is None:
            file_in = self.header_file
        assert file_in is not None

        try:
            with h5py.File(file_in, 'w') as fiout:
                if not fiout.attrs['aares_file_type'] == 'data_file_headers':
                    raise OSError
                if not fiout.attrs['aares_version'] == aares.version:
                    logging.warning('AAres version used and of the file does not match.')

                for name, header in fiout:
                    self.files_dict[name] = h5z.SaxspointH5(header)

        except OSError or KeyError or TypeError:
            raise OSError('File is not of correct format or type: {}'.format(file_in))


