import copy
import glob
import itertools
import math
import os
import logging
import re
import datetime

import freephil as phil

import aares
import h5z
from aares import power as pwr, my_print
import aares.datafiles.data1D
from aares.datafiles.data1D import Reduced1D, Subtract1D

#from aares.import_file import phil_core as import_phil

data_file_types = {'SaxspointH5': h5z.SaxspointH5,
                   #'Reduced1D': aares.datafiles.data1D.Reduced1D,
                   'Reduced1D': Reduced1D,
                   'Subtracted1D': Subtract1D
                   }


def get_file_type(fin):
    '''
    Returns file class suitable for the file
    '''

    for name, tp in data_file_types.items():
        if tp.is_type(fin):
            return tp
            # if name == 'Reduced1D':
            #     base_type_name = aares.datafiles.data1D.Reduced1D.base_class_name(fin)
            #     return aares.datafiles.data1D.Reduced1D
            # else:
            #     return tp
    raise TypeError('Unknown file type')

def read_file(path):
    '''
    Reads file using associtated class
    '''
    file_class = get_file_type(path)
    return file_class(path)

group_phil_str = '''
    name = None
    .expert_level = 0
    .help = Name of the group
    .type = str

    geometry = None
    .type = path
    .help = File with geometry description. Special PHIL file, or H5 with correct header
    .expert_level = 3

    q_space = None
    .type = path
    .help = File with detector transformed to Q space (Output of aares.q_transformation).
    
    phil = None
    .type = path
    .multiple = True
    .help = Processing configuration for this group. Phil file is accepted.

    mask = None
    .type = path
    .help = File with the mask to be used. File of the same form as from aares.mask
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
        
        frames = None
        .type = str
        #.help = "Use only selected frames from the file. The frames are 0-indexed. The format is comma-separated list of frames or frame ranges, range boundaries are column-separated, left is inclusive, eight is exclusive. Valid examples: [1,2,3, 8,9] [:3, 6:9, 12:] [3,4, 8:12]"
        
        is_background = None
        .type = bool
        .help = "Whether the file was recognized as background or not"
        
        background = None
        .type = str
        .help = "Which background file to use. The file is specified by its `file.name`"
    }
'''
group_phil = phil.parse(group_phil_str)
phil_files = phil.parse('''
headers = None
.type = path
.help = "File storing parsed headers. If the file is not present, headers are read from the original"
        "data files."

group 
.multiple = True
.help = Group of files with the same geometry header
.expert_level = 0
{
    ''' + group_phil_str + '''   

}
''')


def validate_hdf5_files(files):
    """
    Checks, if the list of files are valid HDF5. Those, which are not are removed from the list.

    :param files: List of file-paths to be checked
    :type files: List of strings
    """
    assert isinstance(files,list)
    for fi in files:
        if not h5z.is_h5_file(fi):
            logging.warning(f'File is invalid or cannot be read, skipping: {fi}')
            files.remove(fi)

def validate_headers(file_dict):
    '''
    Check, if the file headers are usable. If not, removes from the dict.
    :param file_dict: dictionary of the file headers
    :type file_dict: FileHeadersDictionary
    :return: The dictionary is modified in place
    '''

    for name in list(file_dict.keys()):
        if not file_dict[name].validate():
            logging.info(f'File was removed from processing as invalid: {name}')
            file_dict.pop(name)



def is_background(file_name, pattern):
    '''
    Returns true if `pattern` is present in the `file_name`
    '''

    if not isinstance(pattern, list):
        pattern = [pattern]

    search_patterns = [re.compile(patt) for patt in pattern]

    return any(patt.search(file_name) is not None for patt in search_patterns)

def detect_background(group, pattern, search_in='name'):
    '''
    Detect, if the file is a background file based on `pattern`.
    :param group: group scope_extract
    '''

    for fi in group.file:
        if search_in == 'name':
            search = fi.name
        elif search_in == 'path':
            search = fi.path
        else:
            raise ValueError(f'Invalid `search_in`: {search_in}')
        fi.is_background = is_background(search, pattern)


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
    return [os.path.normpath(fi) for fi in file_list]


def match_headers(hd1, hd2, headers_to_match='entry/instrument/detector'):
    if isinstance(headers_to_match, str):
        headers_to_match = [headers_to_match]

    return all(hd1[match] == hd2[match] for match in headers_to_match)

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
        try:
            if is_merged(hd) and ignore_merged:
                continue

            file_scope = file_object(path=fi)

            for group in groups:
                if match_headers(hd, files[group.file[0].path], headers_to_match):
                    group.file.append(file_scope)
                    break
            else:
                groups.append(group_object([file_scope], name='group{:03d}'.format(group_id)))
                group_id += 1
        except KeyError as e:
            raise KeyError("Error while processing file: {}\n{}".format(fi, repr(e)))

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

class FileGroup():
    """
    Holds information on group of files with common geometry

    :var __group_phil: Phil scope_extract with processing parameters for the group
    :var __scope_extract: group entry in FLS file
    """
    def __init__(self, main_phil, scope_in = None ):
        assert isinstance(scope_in, phil.scope_extract) or scope_in is None
        assert isinstance(main_phil, phil.scope)
        self._main_phil = main_phil
        work_scope = group_phil.fetch(group_phil.format(scope_in))
        self.__group_phil = None

        setattr(self, '_scope_extract', work_scope.extract()) # Has to be the last one

    def __getattr__(self, item):
        if item not in dir(self):
            return getattr(self._scope_extract,item)
        else:
            return getattr(self, item)

    def __setattr__(self, key, value):

        if (key in dir(self)) or ('_scope_extract' not in dir(self)):
            self.__dict__[key] = value
        else:
            setattr(self._scope_extract,key,value)

    def __deepcopy__(self, memodict={}):
        new_copy = FileGroup(copy.deepcopy(self._main_phil), copy.deepcopy(self._scope_extract))
        return new_copy

    @property
    def files_by_name(self):
        return {fi.name: fi for fi in self.file }

    @property
    def group_phil(self):
        '''
        Scope extract of the configuration for the group
        :return:
        '''
        if self.__group_phil is None:
            self.__group_phil = self._main_phil.extract()
            if self.phil is not None:
                for fi in self.phil:
                    if os.path.isfile(fi):
                        fin = phil.parse(file_name=fi)
                        self.__group_phil = self._main_phil.fetch(fin).extract()
                    else:
                        logging.warning('File does not exist, skipping: {}'.format(fi))

        return self.__group_phil

    @group_phil.setter
    def group_phil(self,val):
        self.__group_phil = val

    def update_group_phil(self,phil_in):
        '''
        Updates the groups phil configuration
        :param phil_in:
        :return:
        '''

        current = self._main_phil.format(self.group_phil)

        if isinstance(phil_in,phil.scope_extract):
            phil_in = self._main_phil.format(phil_in)

        updated_scope = self._main_phil.fetch(sources=[current, phil_in]).extract()
        self.__group_phil = updated_scope

    @property
    def geometry(self):
        geom = self._scope_extract.geometry
        if geom is None:
            geom = self.scope_extract.file[0].path
        return geom

    @geometry.setter
    def geometry(self, val):
        self.scope_extract.geometry = val

    def write(self):
        '''
        Writes current work phil configuration to file
        :return:
        '''
        work_scope = self._main_phil.format(self.group_phil)
        i = 0
        while (os.path.isfile(file_name := self.name+'_work'+str(i)+'.phil')):
            i += 1
        self.phil = [file_name] + self.phil

        try:
            with open(file_name, 'w') as fout:
                fout.write(work_scope.as_str())
        except PermissionError:
            raise aares.RuntimeErrorUser('Cannot write to file: {}'.format(file_name))

        return file_name

    @property
    def scope_extract(self):
        return self._scope_extract

    @scope_extract.setter
    def scope_extract(self, val):
        self._scope_extract = val

class File_Groups(list):
    pass
    # def append(self, obj):
    #     if not ((isinstance(obj, phil.scope_extract) or isinstance(obj,FileGroup))
    #                             and hasattr(obj, 'file')):
    #         raise AttributeError('Appended item probably is not group')
    #
    #     if isinstance(obj, phil.scope_extract):
    #         new_obj = FileGroup(self.main_phil, obj)
    #     else:
    #         new_obj = obj
    #
    #     super().append(new_obj)
    #
    # def extend(self, __iterable):
    #     for it in __iterable:
    #         self.append(it)

class DataFilesCarrier:
    """
    Handling class to control file imports

    :ivar files_dict: Dictionary of read file headers. File path as a key.
    :ivar file_groups: Files ordered into the groups (phil.scope_extract)
    :ivar file_scope: Full phil.scope_extract of phil_files
    """

    _master_group_name = '/file_headers'

    def __init__(self, run_phil=None, file_phil=None, nproc=None, mainphil=None):
        """
        :param run_phil:
        :type  run_phil: phil.scope_extract
        :return: Returns a list of files separated to groups in format of `import_file.phil_file`
        :rtype: phil.scope_extract
        """

        assert (run_phil is not None) ^ (file_phil is not None)
        assert isinstance(mainphil, phil.scope)

        self.__header_file = None
        self.main_phil = mainphil
        self._files_dict = FileHeadersDictionary()
        self.file_scope = phil_files.extract()

       # self.__file_groups = File_Groups()
        #self.file_groups = None
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
    def file_scope(self):

        out = self.__file_scope
        out.group = []
        for gr in self.file_groups:
            out.group.append(gr.scope_extract)
        return out

    @file_scope.setter
    def file_scope(self, val):
        self.__file_scope = val
        self.file_groups = val.group

    @property
    def files_dict(self):
        return self._files_dict

    @files_dict.setter
    def files_dict(self,item):
        self._files_dict = FileHeadersDictionary()
        self._files_dict.update(item)

    @property
    def file_groups(self):
      #  return self.file_scope.group
        return self.__file_groups

    @file_groups.setter
    def file_groups(self, item):
        self.__file_groups = File_Groups()
        for val in item:
            self.__file_groups.append(FileGroup(self.main_phil, val))

#        self.file_scope.group = self.__file_groups #item

    @property
    def header_file(self):
        return self.__file_scope.headers

    @header_file.setter
    def header_file(self, item):
        self.__file_scope.headers = item

    def from_input_phil(self, phil_in):
        """
        Processes files using an input parameters in PHIL
        :return:
        """
        from aares.import_file import phil_core
        try:
            phil_core.format(phil_in)
        except : #TODO: chceck what exception can actually occure
            raise AttributeError('Wrong input Phil parameters.')

        files = get_files(phil_in.search_string, phil_in.suffix)
        aares.my_print('Found {} files. Reading headers...'.format(len(files)))

        validate_hdf5_files(files)

        if len(files) == 0:
            raise aares.RuntimeErrorUser('No usable files found.')

        self.files_dict = pwr.get_headers_dict(files, nproc=self.nproc)
        validate_headers(self.files_dict)

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
  #      self.file_groups = phil_files.extract()
        self.file_groups = groups

        if phil_in.headers is None:
            phil_in.headers = os.path.splitext(phil_in.output)[0]+'.hdr'

        self.header_file = phil_in.headers

        self.set_group_geometries()

        self.set_name(phil_in)

    def set_name(self, phil_in,
                 sep='_'):
        if phil_in is None:
            phil_in = self.main_phil.extract().to_import
        if phil_in.prefix == 'time':
            self.sort_by_time()
            prefix = True
        elif phil_in.prefix == 'file_name':
            prefix = True
        else:
            prefix = False

        if phil_in.name_from == "file_name":
            self.set_name_from_filename(strip_common=phil_in.shorten,  sep= sep)
        elif phil_in.name_from == "sample":
            self.set_name_from_sample()
        else:
            raise aares.RuntimeErrorUser('Not sure, what to do. Please set "name_from".')

        if prefix:

            i = 1
            digit = int(math.log10(len(self.files_dict))) + 1
            for fi in self.files_dict.keys():
                scope = self.get_file_scope(fi, key='path')
                scope.name = sep.join([str(i).zfill(digit), scope.name])
                i += 1

    def from_phil_file(self, phil_in):
        """
        Imports the description of groups from the PHIL file
        :param phil_in: Input PHIL parameters
        :type phil_in: string, phil.scope or phil.scope_exract
        :return:
        """

        if isinstance(phil_in, str):
            if os.path.isfile(phil_in):
                try:
                    phil_in = phil.parse(file_name=phil_in)
                except RuntimeError as e:
                    raise aares.RuntimeErrorUser(str(e))
            else:
                raise aares.RuntimeErrorUser('File not found: {}'.format(phil_in))

        if isinstance(phil_in, phil.scope):
            phil_in = phil_files.fetch(phil_in).extract()

        assert isinstance(phil_in, phil.scope_extract)

        self.file_scope = phil_files.format(phil_in).extract()
 #       self.file_groups = self.file_scope.group
        if self.header_file is not None:
            if os.path.isfile(self.header_file):
                logging.debug('Reading pre-extracted headers from: {}'.format(self.header_file))
                self.read_headers_from_file()
            else:
                logging.debug('File with pre-extracted headers ({}) does not exist, reading headers from data files...'.format(self.header_file))
                self.read_headers()
                self.write_headers_to_file()
        else:
            logging.debug('No file with pre-read headers specified in input PHIL. Reading the headers...')
            self.read_headers()

    def update(self,search_string, run_phil=None, suffixes= ['h5z', 'h5'], ignore_merged=True):
        '''
        Updates list of files, and reads their headers, if they don't exist yet.

        :return: Dictionary of the new files
        '''
        files = get_files(search_string, suffixes)
        new_files = [fi for fi in files if fi not in self.files_dict.keys()]

        validate_hdf5_files(new_files)

        if len(new_files) == 0:
            aares.my_print('No new or updated valid files identified.')
            return {}
        logging.debug('Found {} new files to be considered.'.format(len(new_files)))
        logging.debug('Reading headers of the new files...')
        new_files_dict = pwr.get_headers_dict(new_files,nproc=self.nproc)

        validate_headers(new_files_dict)

        new_groups = files_to_groups(new_files_dict, h5z.SaxspointH5.geometry_fields,
                                     ignore_merged=ignore_merged)

        for group in new_groups:
            for old_gr in self.file_scope.group:
                if match_headers(new_files_dict[group.file[0].path],
                                 self.files_dict[old_gr.file[0].path],
                                 h5z.SaxspointH5.geometry_fields):
                    old_gr.file.extend(group.file)
                    break
            else:
                logging.warning('New geometry group introduced. Chechinkg the output files is highly advised, reprocessing might be needed.')
                while group.name in [gr.name for gr in self.file_groups]:
                    m = re.search(r'\d+$', group.name)
                    i = int(m.group())+1 if m else 1
                    group.name = group.name.split(m.group())[0]+'{:03d}'.format(i)
                self.file_scope.group.append(group) # It might be needed to create FileGroup.

        self.set_group_geometries()
        file_list = list(self.files(key='path'))
        new_files_dict_out = {path: header for path, header in new_files_dict.items() if path in file_list}
        self.files_dict.update(new_files_dict_out)

        # Set new names
        self_copy = copy.deepcopy(self)
        self_copy.set_name(run_phil)

        for fi in new_files_dict_out.keys():
            self_fi_sc = self.get_file_scope(fi, key='path')
            copy_fi_sc = self_copy.get_file_scope(fi, key='path')
            self_fi_sc.name = copy_fi_sc.name

        return new_files_dict_out


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

    def set_name_from_sample(self, no_sample='Unknown'):
        """
        Set group.file.name based on the sample name
        """

        for fi, hdr in self.files_dict.items():
            scope = self.get_file_scope(fi, key='path')
            scope.name = hdr.sample_name
            if scope.name is None:
                logging.warning('File does not have sample name set: {}'.format(fi))
                scope.name = no_sample


    def set_name_from_filename(self, strip_common=True, sep='_'):
        """
        Set group.file.name based on the file name

        :param strip_common: Uses only unique part of the name
        :return:
        """
        for group in self.file_groups:
            for fi in group.file:
                fi.name = os.path.abspath(fi.path)

        if strip_common:
            for group in self.file_groups:
                #Strip file path
                if len(group.file) == 1:
                    group.file[0].name = os.path.splitext(os.path.split(group.file[0].path)[1])[0]
                else:
                    filepaths = [os.path.splitext(fi.name)[0] for fi in group.file]
                    common_path = os.path.commonpath(filepaths)
                    for fi in group.file:
                        if len(common_path) > 0:
                            fi.name = fi.name.split(common_path)[1].strip('.\/_'+sep+os.sep)
                        else:
                            fi.name = fi.path
                        fi.name = os.path.splitext(fi.name)[0].replace(os.sep,'_')

                    common = longest_common([fi.name for fi in group.file])
                    for fi in group.file:
                        name = fi.name.split(sep)
                        for cmn in common:
                            name.remove(cmn)
                        fi.name = sep.join(name)

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

    def files(self, key='name'):
        """
        Iterates over file keys
        :param key: Key to iterate over
        :return: keys
        """
        if not self._is_file_key(key):
            raise AttributeError('This key type is not used: {}'.format(key))

        for gr in self.file_scope.group:
            for fi in gr.file:
                yield fi.__dict__[key]

    def read_headers(self):
        """
        Reads the file headers, and fills the self.files_dict
        """
        self.files_dict = pwr.get_headers_dict(list(self.files('path')), nproc=self.nproc)
        validate_headers(self.files_dict)

    def sort_by_time(self):
        """
        Sort the file dictionary by header time
        """
        sorted_dict = {k: self.files_dict[k] for k in sorted(self.files_dict,
                                                             key=lambda x:self.files_dict[x].file_time_iso)}

        self.files_dict = sorted_dict

    def detect_background(self, pattern='buffer', search_in='name'):
        for group in self.file_groups:
            detect_background(group, pattern, search_in=search_in)

    def write_groups(self,file_out='files.fls', update=True):
        '''
        Write the file groups to a file
        :return:
        '''

        # if update:
        #     self.file_scope.group = []
        #     for group in self.file_groups:
        #         self.file_scope.group.append(group.scope_extract)

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

        master_group_name = self._master_group_name

        if file_out is None:
            file_out = self.header_file
        assert file_out is not None

        master_group = h5z.GroupH5(name='/')
        master_group[master_group_name] = h5z.GroupH5(name=master_group_name)

        i = 0
        digit = int(math.log10(len(self.files_dict))) + 1
        for name, header in self.files_dict.items():
            key =str(i).zfill(digit)
            header_out = copy.deepcopy(header)
            master_group[master_group_name + '/' + key] = header_out._h5
            entry = master_group[master_group_name + '/' + key]
            entry.name = master_group_name + '/' + key
            for it_key in entry.walk():
                new_key = master_group_name + '/' + key +  entry[it_key].name
                entry[it_key].name = new_key


            i += 1

        master_group.attrs['aares_file_type'] = 'data_file_headers'
        master_group.attrs['aares_version'] = str(aares.version)

        try:
            with h5py.File(file_out, 'w') as fiout:
                master_group.write(fiout, compression='gzip')
        except PermissionError:
            aares.RuntimeErrorUser('Cannot write to {}. Permission denied.'.format(file_out))

    def read_headers_from_file(self, file_in: object = None) -> object:
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
            with h5py.File(file_in, 'r') as fiin:
                if not fiin.attrs['aares_file_type'] == 'data_file_headers':
                    raise OSError
                if not fiin.attrs['aares_version'] == str(aares.version):
                    logging.warning('AAres version used and of the file does not match.')

                input_group = h5z.GroupH5(fiin)

            for key in input_group[self._master_group_name].walk():
                assert key[0] == '/'
                split_name = input_group[key].name.split('/')
                input_group[key].name = '/' + '/'.join(split_name[3:])

        except OSError or KeyError or TypeError:
            raise OSError('File is not of correct format or type: {}'.format(file_in))

        for item in input_group[self._master_group_name].values():
            header = h5z.SaxspointH5(item)
            self.files_dict[header.path] = header

    def assign_background(self, method='time'):
        '''
        Assign background file to each file.

        :param files: Files to be processed
        :type files: DataFilesCarrier
        '''

        if method == 'time':
            for group in self.file_groups:
                my_print('Assigning background files for group {}'.format(group.name))
                buffers = {fi.name: datetime.datetime.fromisoformat(self.files_dict[fi.path].file_time_iso)
                           for fi in group.file if fi.is_background}
                if len(buffers) == 0:
                    logging.warning('No background files found for this group.')
                    continue
                logging.info('Found {} background files.'.format(len(buffers)))
                for fi in group.file:
                    if fi.is_background:
                        continue
                    file_time = datetime.datetime.fromisoformat(self.files_dict[fi.path].file_time_iso)
                    for buff, tm in buffers.items():
                        if tm < file_time:
                            best_buffer = buff
                            buffer_time = tm
                            break
                    else:
                        logging.warning('No suitable background for this file: {}'.format(fi.name))
                        continue

                    for buff, tm in buffers.items():
                        if buffer_time < tm < file_time:
                            best_buffer = buff
                            buffer_time = tm

                    fi.background = best_buffer

        else:
            raise NotImplementedError('Unknown method')