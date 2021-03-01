"""
Importing data files


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import aares
import logging

__all__ = []
__version__ = aares.__version__
prog_short_description = 'Finds and import the data files'

import h5z
import freephil as phil
import glob
import os
import itertools
import aares.power as pwr

phil_files = phil.parse('''
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

phil_core = phil.parse('''
to_import {
search_string = None
.help = How should be searched for the file. If it is a folder, it is searched recursively for all data files. Bash-like regex strings are accepted.
.multiple = True
.type = str

suffix = h5z h5
.help = Suffixes to search for. HDF5 files are used by the software.


output = files.phil
.help = PHIL file with description of the imported files and file groups.
.type = path

name_from = None sample *file_name
.help = How to generate name of the file
.type = choice

shorten = True
.help = Strip the common parts of the file name
.type = bool

prefix = None *time file_name
.help = Prefix the name enumerated by...
.type = choice

ignore_merged = True
.help = Ignore files with merged frames. Using these diminish some of the aares features. Moreover, no special handlig fo these is implemented, which can lead to unexpected results.
.type = bool
.expert_level = 1
}
''')


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
        file_in['entry/data/averaged_frames']
        return True
    except KeyError:
        return False


class ImportFiles:
    """
    Handling class to control file imports

    :ivar files_dict: Dictionary of read file headers. File path as a key.
    :ivar file_groups: Files ordered into the groups (phil.scope_extract)
    """

    def __init__(self, run_phil=None, file_phil=None, nproc=None):
        """
        :param run_phil:
        :type  run_phil: phil.scope_extract
        :return: Returns a list of files separated to groups in format of `import_file.phil_file`
        :rtype: phil.scope_extract
        """

        assert (run_phil is not None) ^ (file_phil is not None)

        self.files_dict = {}
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

        self.file_groups = phil_files.format(phil_in).extract().group
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

    def get_file_scope(self, value, key='name'):
        """
        Searches self.file_groups for a file, whose key match value. Returns respective file scope.

        :param value: Description of the file to be searched for.
        :param key: A key name to be searched in. Has to be one of group.file keys
        :rtype: phil.scope_extract
        """

        if self._is_file_key(key):
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


    def write_groups(self,file_out='files.phil'):
        '''
        Write the file groups to a file
        :return:
        '''

        try:
            group_out = phil_files.extract()
            group_out.group = self.file_groups
            with open(file_out, 'w') as fiout:
                phil_files.format(group_out).show(out=fiout)
        except PermissionError:
            aares.RuntimeErrorUser('Cannot write to {}. Permission denied.'.format(fiout))


class JobImport(aares.Job):
    """
    Run class based on generic AAres run class
    """

    def __set_meta__(self):
        '''
        Sets various package metadata
        '''
        import os, sys
        self._program_short_description = prog_short_description

        self._program_name = os.path.basename(sys.argv[0])
        self._program_version = __version__

    def __worker__(self):
        '''
        The actual programme worker
        :return:
        '''

        run = ImportFiles(self.params)

        files = phil_files.format(run.file_groups)
        #       print(files.as_str(expert_level=0))
        if self.params.output is not None:
            run.write_groups(self.params.output)

            aares.my_print('List of imported files was written to: {}'.format(self.params.output))

        pass

    def __set_system_phil__(self):
        '''
        Settings of CLI arguments. self._parser to be used as argparse.ArgumentParser()
        '''
        self._system_phil = phil_core

    def __argument_processing__(self):
        '''
        Adjustments of raw input arguments. Modifies self._args, if needed

        '''
        pass

    def __process_unhandled__(self):
        '''
        Process unhandled CLI arguments into self.params

        :return:
        '''
        self.params.search_string.extend(self.unhandled)

    def __help_epilog__(self):
        '''
        Epilog for the help

        '''
        pass


def main():
    job = JobImport()
    return job.job_exit


if __name__ == '__main__':
    main()
