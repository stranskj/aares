import time, datetime
import sys, os
import argparse
import traceback
import logging
import logging.config
from .version import version
import freephil as phil

__version__ = version

__license__ = '''
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


class RuntimeErrorUser(RuntimeError):
    '''
    Exception marking user-related errors, which should be handled user-friendly.
    '''
    pass


class RuntimeWarningUser(RuntimeError):
    '''
    Exception marking user-related warnings, which should be handled user-friendly.
    '''
    pass


def logging_config(prog_name='ares'):
    return dict(
        version=1,
        formatters={
            'simple_print': {'format': '%(message)s'},
            'with_severity': {'format': '\n%(levelname)s: %(message)s'}
        },
        handlers={
            'print_stdout': {
                'class': 'logging.StreamHandler',
                'formatter': 'with_severity',
                'level': logging.WARNING
            },
            'file_log': {
                'class': 'logging.FileHandler',
                'formatter': 'simple_print',
                'level': logging.INFO,
                'filename': 'ares.log',
                'mode': 'a'
            },
            'debug_file': {
                'class': 'logging.FileHandler',
                'formatter': 'simple_print',
                'level': logging.DEBUG,
                'filename': prog_name + '.debug.log',
                'mode': 'w'
            },
        },
        root={
            'handlers': ['print_stdout', 'file_log', 'debug_file'],
            'level': logging.DEBUG
        }

    )


def my_print(msg):
    '''
    Prints to sdout, but also logs as logging.INFO
    :param args:
    :return:
    '''

    print(msg)
    logging.info(msg)


def run_task(task, *args, msg=None, **kwargs):
    '''
    Runs function, and reports on its progress to stdout
    '''
    print(msg, end='')
    try:
        start_t = time.time()
        result = task(*args, **kwargs)
        el_time = time.time() - start_t
    except Exception as e:
        print('FAILED')
        raise
    else:
        print('{:.1f} s'.format(el_time))
        return result

def search_files(indir, suffix):
    fiout = []
    for root, dirs, files in os.walk(indir):
        for file in files:
            if file.endswith(suffix):
                fiout.append(os.path.join(root, file))
    return fiout

class Job(object):
    '''
    Generic class for running a programme. Actual programmes inherits this class and modifies class members appropriately.
    '''

    def __init__(self, *args, run=True, logger=None, **kwargs):
        '''
        Runs the programme using class members. Args, kwargs are passed over to the main programme function.
        '''
        self._args = 'Arguments not parsed yet.'
        Job.__set_meta__(self)
        self._logger = logger or logging.config.dictConfig(logging_config(self._program_name))

        self.__set_meta__()
        self.__set_argument_parser__()
        self._system_phil = phil.parse('')
        self.__set_system_phil__()
        self.job_exit = None
        self._args = argparse.Namespace()
        self._args.verbosity = 3  # Ensures printing exception callback, if something goes wrong before running the __worker__
        if run:
            self.__run__(*args, **kwargs)


    def __set_meta__(self):
        '''
        Sets various package metadata
        '''

        self._program_short_description = 'Another angular REduction for Saxs'

        self._program_name = os.path.basename(sys.argv[0])
        self._program_version = __version__

    def __worker__(self):
        '''
        The actual programme worker
        :return:
        '''
        pass

    def __set_system_phil__(self):
        '''
        Settings of CLI arguments. self._parser to be used as argparse.ArgumentParser()
        '''
        pass

    def __argument_processing__(self):
        '''
        Adjustments of raw input arguments. Modifies self._args, if needed

        '''
        pass

    def __help_epilog__(self):
        '''
        Epilog for the help

        '''
        pass

    def __set_argument_parser__(self, parser=None, **kwargs):
        if parser is not None:
            self._parser = parser
        else:
            description = '''{short_desc}
            
USAGE
'''.format(short_desc=self._program_short_description)
            self._parser = argparse.ArgumentParser(description=description,
                                                   #            formatter_class=argparse.RawDescriptionHelpFormatter,
                                                   epilog=self.__help_epilog__(),
                                                   **kwargs)
            program_version_message = '%%(prog)s %s \nPython %s' % (self._program_version, sys.version)
            self._parser.add_argument('-V', '--version',
                                  action='version',
                                  version=program_version_message)
            self._parser.add_argument('-c','--show-config',
                                      nargs='?',
                                      const='True',
                                      default=None,
                                      dest='show_config',
                                      help='Show the configuration parameters. If file name is provided, generates PHIL file')
            self._parser.add_argument('-a','--attributes-level',
                                      default=0,
                                      type=int,
                                      dest='attributes_level',
                                      help='Set the attributes level for showing configuration parameters')
            self._parser.add_argument('-e', '--expert-level',
                                      default=0,
                                      type=int,
                                      dest='expert_level',
                                      help='Set the extra level for showing configuration parameters')
            self._parser.add_argument('--phil',
                                      action="append",
                                      metavar="FILE",
                                      help="PHIL files to read. Pass '-' for STDIN. Can be specified multiple times, but duplicates ignored.",
                                      )
            self._parser.add_argument('-v', '--verbosity',
                                      dest='verbosity',
                                      default=0,
                                      action='count',
                                      help='Increases output verbosity')
            self._parser.add_argument('phil_arguments',
                                      nargs=argparse.REMAINDER)

    def __parse_arguments__(self,argv=None):
        '''

        :param argv:
        :return:
        '''
        if argv is None:
            argv = sys.argv[1:]

        self._args = self._parser.parse_args(argv)


        self.__argument_processing__()

    def phil_argument_processing(self, args, verbose=False, return_unhandled=False, quick_parse=False):
        '''
        Processes phil commands from CLI
        :return:
        '''

        def _is_a_phil_file(filename):
            return any(
                filename.endswith(phil_ext)
                for phil_ext in (".phil", ".param", ".params", ".eff", ".def")
            )

        # Parse the command line phil parameters
        user_phils = []
        unhandled = []
        interpretor = self.system_phil.command_line_argument_interpreter()

        #TODO: enable space separated = ; clever splits and strips?

        for arg in args:
            if (
                _is_a_phil_file(arg)
                and os.path.isfile(arg)
                and os.path.getsize(arg) > 0
            ):
                try:
                    user_phils.append(phil.parse(file_name=arg))
                except Exception:
                    if return_unhandled:
                        unhandled.append(arg)
                    else:
                        raise
            # Treat "has a schema" as "looks like a URL (not phil)
            elif "=" in arg: #and not urlparse(arg).scheme:
                try:
                    user_phils.append(interpretor.process_arg(arg=arg))
                except Exception:
                    if return_unhandled:
                        unhandled.append(arg)
                    else:
                        raise
            else:
                unhandled.append(arg)

        # Fetch the phil parameters
        self._phil, unused = self.system_phil.fetch(
            sources=user_phils, track_unused_definitions=True
        )

        # Print if bad definitions
        if len(unused) > 0:
            msg = [item.object.as_str().strip() for item in unused]
            msg = "\n".join(["  %s" % line for line in msg])
            raise RuntimeErrorUser(
                "The following definitions were not recognised\n%s" % msg
            )

        # Extract the parameters
        self.params = self._phil.extract()
        self.unhandled = unhandled

    def __run__(self,*args,**kwargs):
        '''
        Running the programme
        '''

        try:

            self.__parse_arguments__()
            self.__intro__()

            if self._args.show_config is not None:
                print(
                "Showing configuration parameters with:\n"
                "  attributes_level = %d\n"
                "  expert_level = %d\n"
                % (self._args.attributes_level, self._args.expert_level)
                )
                print(
                    self.system_phil.as_str(
                        expert_level=self._args.expert_level,
                        attributes_level=self._args.attributes_level,
                    )
                )
                if not (self._args.show_config == 'True'):
                    print('Writing to file: {}'.format(self._args.show_config))
                    try:
                        with open(self._args.show_config, 'w') as conf_out:
                            self.system_phil.show(
                                conf_out,
                                expert_level=self._args.expert_level,
                                attributes_level=self._args.attributes_level,)
                    except:
                        raise RuntimeErrorUser('Cannot write to file {}'.format(self._args.show_config))

                self.job_exit = 0
                return

            self.phil_argument_processing(self._args.phil_arguments)

            start_t = time.time()
            self.__worker__()
            el_time = time.time() - start_t
            my_print('\nFinished. Time elapsed: {:.1f} s'.format(el_time))
            self.job_exit = 0

        except KeyboardInterrupt:
            my_print('Keyboard interrupt. Exiting...')
            self.job_exit = 0

        except RuntimeErrorUser as e:
            logging.error('ERROR: ' + str(e))
            self.job_exit = 2

        except Exception as e:
            logging.debug('''
Program name:    {prog_name}
Package version: {prog_ver}
Python version:  {python_ver}
Program input:
{cli_in}
Input understanding:
{args}'''.format(prog_name=self._program_name,
                 prog_ver=self._program_version,
                 python_ver=sys.version,
                 cli_in=sys.argv,
                 args=self._args))

            logging.exception(e)
            print('\nPlease, report back to the developers. Include the debug file with the report: {}'.format(self._program_name+'.debug.log'))
            self.job_exit = 2

    def __intro__(self):
        my_print("")
        my_print("\t========================")
        my_print('\t{}'.format(self._program_name))
        my_print("\tVersion: {}".format(__version__))
        my_print("\tDate: {}".format(datetime.datetime.now()))
        my_print("\tAuthor: Jan Stransky")
        my_print("\t========================")
        logging.debug('Python and system versions:\n' + sys.version)
        my_print(" ")
        my_print(__license__)
        my_print(" ")
        logging.info('Run command:\n' + " ".join(sys.argv) + '\n')

    @property
    def phil(self):
        '''
        Get phil object
        :return:
        '''

        return self._phil

    @property
    def system_phil(self):
        '''
        Get system phil object
        :return:
        '''

        return self._system_phil

    @property
    def diff_phil(self):
        '''
        Get the diff phil
        :return: The difference phil scope
        '''