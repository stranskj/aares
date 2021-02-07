import ares
import sys,os
import freephil as phil

__all__ = []
__version__ = ares.__version__
prog_short_description = 'Integrates scattered intensity'

phil_scope = phil.parse(
    '''
test.output = None
   .type = str
   .help = "Testing parametr for output"
   .expert_level = 0

test.input = None
   .type = str
   .help = "testing input"  
   .expert_level = 3  


paths = None
    .type = str
    .help = "Test"
    .multiple=True

    '''


)

class JobTestPhil(ares.Job):
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
        self._system_phil = phil_scope

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

    def __process_unhandled__(self):
        '''
        Process unhandled CLI arguments into self.params

        :return:
        '''
        self.params.paths=self.unhandled

def main(argv=None):
    job = JobTestPhil()
    return job.job_exit


if __name__ == "__main__":
    sys.exit(main())