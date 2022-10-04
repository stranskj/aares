"""
Angular reduction


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import h5z
import h5py
import numpy
import math
import aares
import aares.power as pwr
import aares.datafiles
import aares.q_transformation
import aares.mask
import aares.export

import concurrent.futures
import os, logging
import freephil as phil

prog_short_description = 'Performs data reduction from 2D to 1D.'

phil_core_str = '''
reduction
.help = Parameters controlling the reduction from 2D to 1D
 {
    bins_number = None
    .type = int
    .expert_level = 0
    .help = Number of bins to which data are divided. If 0 or None, determined automatically.
    
    q_range = 0 0
    .type = floats(2)
    .expert_level = 0
    .help = Q range to be used for the reduction
   
    beam_normalize
    .help = Adjust data for beam variation. Pick one of the options q_range or real_space. If 0, it is not used. 
    {
        q_range = None
        .type = floats(2)
        .expert_level = 0
        .help = Adjust to the mean within the q-range. For example, it can be used for backgraound normalization.
        
        real_space = None
        .type = floats(2)
        .expert_level = 0
        .help = Adjust to the mean within the real distance on the detector. Useful for normalization using semi-transparent beamstop: use balue slightly smaller than the beamstop size
        
        beamstop_transmission = 1
        .type = float
        .expert_level = 1
        .help = Transmission constant of the semitransparent beamstop (Not implemented yet).
        
        scale = None
        .type = float
        .expert_level=1
        .help = Value, to which all the frames are scaled. If None, value from the first frame is chosen in such a way, that the scale of that frame is 1. (not implemented yet)
    }
    
    transmitance = None
    .type = bool
    .expert_level = 0
    .help = Normalize the data to transmitance, based on flux measurements. If None, applied if the flux data are available in the data.
    
    file_bin_masks = None
    .type = path
    .help = File containing binning and reduction masks (output of previous aares.integrate).
    .expert_level=1
    
    error_model = *3d pixel poisson
    .type = choice
    .help = 'Model used to estimate measurement errors for intensity.'
            ' 3d - Standard deviation of intensity at all pixels with given q (within g-bin).'
            ' pixel - standard deviation at pixel across all frames followed by error propagation.'
            ' poisson - standard deviation from Poisson distribution (e.g. square root of intensity).'
    
    empty_skip = True
    .type = bool
    .help = If the bin contains zero pixels, it is ignored.

}     
'''

phil_core = phil.parse(phil_core_str)

phil_job_core = phil.parse('''
    include scope aares.common.phil_input_files
    
    input {
    file_name = None
    .multiple = True
    .type = path
    .help =  Files with the data to be processed. It needs to be explicit file name. Use "aares.import" for more complex file search and handling.
    
    q_space = None
    .type = path
    .help = File containing Q-transformation information (output of aares.q_transformation)
    
    mask = None
    .type = path
    .help = PNG file with mask (output from aares.mask)
    }
    
    export {
        separator = comma *space semicolon
        .type = choice
        .optional = False
        .help = Separator to be used in the output file
        file_name = original *name sample
        .type = choice
        .optional = False
        .help = "How should be derived file name for the exported data. 
                'original' - same as the source file name; 
                'name' - same as the name in the AAres imported files;
                'sample' - as sample name specified in data file header. A number is preceeded to guarantee uniquenes. No order is guaranteed."
    }
    
    
    output {
        directory = 'reduced'
        .type = path
        .help = Output folder for the processed data
        input_files = binned.fls
        .type = path
        .help = Updated descriptor of the input files.
    }

    ''' + phil_core_str + '''
    
    include scope aares.power.phil_job_control
''', process_includes=True)


def create_bins(qmin, qmax, bins):
    '''
    Creates list of bins
    :param qmin: float
    :param qmax: float
    :param bins: int
    :return:
    '''

    out = []
    step = math.fabs(qmin - qmax) / bins

    for i in range(bins):
        out.append((qmin + i * step, qmin + (i + 1) * step))

    return out


def integration_mask(bin, q_array, frame_mask=None):
    """
    Creates integration mask for a single bin
    :param q_array: numpy.array
    :param bin: tuple
    :param frame_mask:
    :return:
    """

    bin = sorted(bin)

    q_mask = numpy.logical_and(bin[0] <= q_array, q_array < bin[1])

    if frame_mask is None:
        frame_mask = True

    return numpy.logical_and(q_mask, frame_mask)


def beam_bin_mask(q_range=None, real_space=None, arrQ=None, pixel_size=None):
    """
    Creates an integration mask for scaling beam variation. For example, it should be location of
    primary beam, when semitransparent beamstop is used, or rackground range, where signally only
    from buffer is expected.

    :param q_range, real_space: Range to be used for mean number to which the data should be normalized
    :type q_range, real_space: tuple
    :param arrQ: array of Q-values
    :return: Integration mask
    """

    assert (q_range is not None or real_space is not None) and not (
            q_range is not None and real_space is not None)

    bin_mask = arrQ == False
    if real_space is not None:
        assert pixel_size is not None
        import aares.mask
        beam_xy = numpy.unravel_index(numpy.argmin(arrQ, axis=None), arrQ.shape)
        beamstop_pix_size = sorted((real_space[0] / pixel_size, real_space[1] / pixel_size))

        inner = aares.mask.beamstop_hole(beam_xy, beamstop_pix_size[0], bin_mask)
        outer = aares.mask.beamstop_hole(beam_xy, beamstop_pix_size[1], bin_mask)

        bin_mask = numpy.logical_and(outer, numpy.logical_not(inner))
    else:
        bin_mask = integration_mask(q_range, arrQ)

    return bin_mask


def list_integration_masks(q_bins, q_array, frame_mask=None):
    '''
    Returns list of masks to be used for the integration.

    :param q_array: Numpy array of the frame shape with q-values of the
    :param q_bins: numpy.array of pairs
    :return:
    '''

    no_bins = len(q_bins)

    q_masks = pwr.map_th(integration_mask, q_bins, [q_array] * no_bins, [frame_mask] * no_bins)

    return list(q_masks)


def integrate(frame_arr, bin_masks):
    '''
    Calculate averages and stddevs across frames in all bins
    :param frame_arr: data; 3d np.array
    :param bin_masks: bin masks
    :return: np.array, np. array, np.array: averages, stddevs, number of points in bin
    '''

    if isinstance(bin_masks, numpy.ndarray) and len(bin_masks.shape) == 2:
        bin_masks.shape = (1, *bin_masks.shape)

    averages = []
    stdev_3d = []
    num = []
    num_not_masked = []
    frame_mask = numpy.any(bin_masks,axis=0)
    pixel_stdev = numpy.nanstd(frame_arr, axis=0, dtype='float64')
    stdev_by_pixel = []
    stdev_sqrtI = []

    for binm in bin_masks:
        #   int_mask = numpy.array([binm]*no_frame)
        binval = frame_arr[:, binm]
        not_masked = binval<0
        num_not_masked.append(numpy.sum(not_masked))
 #       binval = binval[binval >= 0]  # TODO: performance hit needs checking; do we need it, e.g. isn't it masked? Maybe we could do it on whole file? Or warn, that there is an masking issue?
        if binval.size <= 0:
            averages.append(numpy.nan)
            stdev_3d.append(numpy.nan)
            stdev_by_pixel.append(numpy.nan)
            stdev_sqrtI.append(numpy.nan)
        else:

            averages.append(numpy.nanmean(binval, dtype='float64'))
            stdev_3d.append(numpy.nanstd(binval, dtype='float64') / math.sqrt(binval.size))
            stdev_by_pixel.append(numpy.nanmean(pixel_stdev[binm], dtype='float64')/math.sqrt(binval.size))
            stdev_sqrtI.append(numpy.sqrt(averages[-1])/math.sqrt(binval.size))

        num.append(binval.size)

    # int_masks = [numpy.array([msk]*no_frame) for msk in bin_masks]

    # averages = [numpy.average(frame_arr[binm]) for binm in int_masks]
    # stdev    = [numpy.std(frame_arr[binm]) for binm in int_masks]

    #    averages = map(numpy.average, [frame_arr]*len(bin_masks), int_masks)
    #    stdev = map(numpy.std,  [frame_arr] * len(bin_masks), int_masks)

    logging.debug('Number of unmasked pixels: {}'.format(numpy.sum(num_not_masked)))
    errors = numpy.array([stdev_3d,
                          stdev_by_pixel,
                          stdev_sqrtI])
    return numpy.array(averages), errors, numpy.array(num)#, numpy.sum(num_not_masked)


def integrate_mp(frame_arr, bin_masks, nproc=None):
    '''
    Calculate averages and stddevs across frames in all bins, parallel in multiple chunks
    :param frame_arr: data; 3d np.array
    :param bin_masks: bin masks
    :return: np.array, np. array: averges and stdevs
    '''

    if nproc is None:
        nproc = os.cpu_count()

    if len(frame_arr.shape) == 2:
        frame_arr.shape = (1, *frame_arr.shape)

    with concurrent.futures.ThreadPoolExecutor(nproc) as ex:
        results = ex.map(integrate, [frame_arr] * nproc,
                         pwr.chunks(bin_masks, int(len(bin_masks) / nproc) + 1))
        # results = pwr.map_th(integrate, [frame_arr]*len(bin_masks), bin_masks, nchunks=nproc)
        resl = list(results)
      #  res = numpy.concatenate(resl, axis=1)

        averages = numpy.concatenate([res[0] for res in resl])
        stdev = numpy.concatenate([res[1] for res in resl], axis=1)
        num = numpy.concatenate([res[2] for res in resl])

  #      print('Number of unmasked pixels: {}'.format(numpy.sum(res[3, :])))

    return averages, stdev, num


def process_file(header, file_out, frames=None, export=None, reduction = None,
                 bin_masks=None,
                 q_val=None,
                 scale=None,
                 scale_transmitance=False,
                 error_model='3d',
                 nproc=None):

    aares.my_print(header.path)

    if frames is None:
        data = header.data
    else:
        try:
            data = aares.slice_array(header.data, intervals=frames, axis=0)
            logging.info('Only {} frames were used from: {}'.format(numpy.size(data, axis=0), header.path))
        except IndexError as err:
            raise aares.RuntimeErrorUser(repr(err)+'\nError while processing file: {}\nCould not select specified frames. Note that frame indices are 0-based.'.format(header.path))

    averages, stddev, num = integrate_mp(data, bin_masks=bin_masks, nproc=nproc)
    if scale is not None:
        frame_scale = scale / averages[-1]
        averages = averages[:-1] * frame_scale
        stddev = stddev[:-1] * abs(frame_scale)
        # avr = avr[:-1]
        # std = std[:-1]
        num = num[:-1]

    if scale_transmitance:
        transmitance = header.transmitance
        averages *= transmitance
        stddev *= transmitance

    error_model = reduction.error_model
    #pick of stddev
    if error_model == '3d':
        stddev = stddev[0]
    elif error_model == 'pixel':
        stddev = stddev[1]
    elif error_model == 'poisson':
        stddev = stddev[2]
    else:
        raise aares.RuntimeErrorUser('Unknown error model: {}'.format(error_model))

    aares.export.write_atsas(q_val, averages,stddev,
                             file_name=file_out,
                             header=['# {}\n'.format(header.path)])


def integrate_group(group, data_dictionary, job_control=None, output=None, export=None, reduction=None):
    '''
    Integrates group of files

    :param group: Group of files to be processed
    :type group: aares.datafiles.FileGroup
    :param data_dictionary: Dictionary with the file headers.
    :type data_dictionary:

    '''

    if job_control is None:
        job_control = pwr.phil_job_control.extract().job_control
        job_control.threads, job_control.jobs = pwr.get_cpu_distribution(job_control)

    params = group.group_phil

    normalize_beam = (params.reduction.beam_normalize.real_space is not None
                      or
                      params.reduction.beam_normalize.q_range is not None)
    if normalize_beam and (params.reduction.beam_normalize.real_space is not None
                           and
                           params.reduction.beam_normalize.q_range is not None):
        raise aares.RuntimeErrorUser('Please provide only one '
                                     'of:\nintegrate.beam_normalize.real_space\nintegrate'
                                     '.beam_normalize.q_range')

    if params.reduction.transmitance is None:
        if all([data_dictionary[fi.path].transmitance is not None for fi in group.scope_extract.file]):
            aares.my_print('Transmitance data found, performing the scaling.')
            scale_transmitance = True
        else:
            scale_transmitance = False
            aares.my_print('Transmitance data not available')
    else:
        scale_transmitance = params.reduction.transmitance
        if scale_transmitance:
            if not all([data_dictionary[fi.path].transmitance is not None for fi in group.scope_extract.file]):
                logging.warning('The data in the group does not contain information for transmitance scaling. Switchin the scaling off.')
                scale_transmitance = False
  #  if not normalize_beam and params.mask.beamstop.semitransparent is not None: TODO: Check and think
  #      normalize_beam = True
  #      params.reduction.beam_normalize.real_space = [0, params.mask.beamstop.semitransparent]

    if params.reduction.beam_normalize.real_space is not None:
        params.reduction.beam_normalize.real_space = numpy.array(
            params.reduction.beam_normalize.real_space) * 0.001 / 2

    aares.my_print('Reading bin masks: {}'.format(params.reduction.file_bin_masks))
    bin_masks_obj = ReductionBins(params.reduction.file_bin_masks)
    aares.my_print('Reading Q-space data: {}'.format(group.scope_extract.q_space))
    arrQ = aares.q_transformation.ArrayQ(group.scope_extract.q_space)
    if normalize_beam:
        aares.my_print('Normalization to beam fluctuation will be performed.')

        beam_mask = beam_bin_mask(
            real_space=params.reduction.beam_normalize.real_space,
            q_range=params.reduction.beam_normalize.q_range,
            arrQ=arrQ.q_length,
            pixel_size=data_dictionary[group.file[0].path].pixel_size[0])

        bin_masks = numpy.append(bin_masks_obj.bin_masks, [beam_mask], axis=0)

        if params.reduction.beam_normalize.scale is None:
            scale, err, num = integrate(data_dictionary[group.file[0].path].data, numpy.array([beam_mask]))
            params.reduction.beam_normalize.scale = scale[0]
            aares.my_print('Normalization scale set to: {:.3f}'.format(scale[0]))
    else:
        bin_masks = bin_masks_obj.bin_masks
    aares.my_print('Using error model: {}\n'.format(reduction.error_model))

    aares.create_directory(output.directory)
    # if not os.path.isdir(output.directory):
    #     try:
    #         os.mkdir(output.directory)
    #     except OSError:
    #         raise aares.RuntimeErrorUser('Path is not a directory: {}'.format(output.directory))

    aares.my_print('Reducing files of {}:'.format(group.scope_extract.name))
    from functools import partial
    process_partial = partial(process_file,
                              export=export,
                              bin_masks=bin_masks,
                              q_val=bin_masks_obj.q_axis,
                              scale=params.reduction.beam_normalize.scale,
                              scale_transmitance=scale_transmitance,
                              reduction=reduction
                              )

    files = [data_dictionary[fi.path] for fi in group.scope_extract.file]
    files_out = [os.path.join(output.directory, fi.name + '.dat')
                 for fi in group.scope_extract.file]  # TODO: use info from export or so

    frames = [fi.frames for fi in group.scope_extract.file]
    aares.power.map_mp(process_partial,
                       files,
                       files_out,
                       frames
                       )


def prepare_bins(arrQ, qmin=None, qmax=None, bins=None, frame_mask=None, skip_empty=True):
    """
    High level function, which prepares q_bins and integration masks

    :param arrQ: Q-transformed frame e.g. array, which holds q-value for each pixel
    :type arrQ: np.array
    :param qmin: q_min value
    :type qmin: float
    :param qmax: q_max value
    :type qmax: float
    :param bins: number of bins
    :type bins: int
    :param frame_mask: Additional mask for the frame.
    :type frame_mask: np.array(bools)
    :param skip_empty: If the bin contains zero pixels, the bin is not created.
    :type skip_empty: Bool
    :return: Returns array of q_values, and corresponding list of masks used for the integration
    """

    import aares.q_transformation as qt

    if frame_mask is None:
        frame_mask = arrQ > 0

    if qmin is None or qmin == 0:
        qmin = arrQ[frame_mask].min()
    if qmax is None or qmax == 0:
        qmax = arrQ[frame_mask].max()

    if (bins is None) or (bins == 0):
        origin = numpy.unravel_index(numpy.argmin(arrQ, axis=None), arrQ.shape)
        to_corners = [math.sqrt((origin[0] - 0) ** 2 + (origin[1] - 0) ** 2),
                      math.sqrt((origin[0] - arrQ.shape[0]) ** 2 + (origin[1] - 0) ** 2),
                      math.sqrt((origin[0] - 0) ** 2 + (origin[1] - arrQ.shape[1]) ** 2),
                      math.sqrt(
                          (origin[0] - arrQ.shape[0]) ** 2 + (origin[1] - arrQ.shape[1]) ** 2)]
        bins = int(max(to_corners) * 0.7)

    q_bins = create_bins(qmin, qmax, bins)
    q_masks = list_integration_masks(q_bins, arrQ, frame_mask)

    if skip_empty:
        bins_prune = []
        mask_prune = []
        skipping = 0
        for bin, mask in zip(q_bins, q_masks):
            if numpy.any(mask):
                bins_prune.append(bin)
                mask_prune.append(mask)
            else:
                skipping += 1

        q_bins = bins_prune
        q_masks = mask_prune
        if skipping > 0:
            logging.warning('Empty bins ({}) were encountered and skipped.'.format(skipping))

    return qt.get_q_axis(q_bins), q_masks


class ReductionBins(h5z.SaxspointH5):
    '''
    Integration bins class
    '''

    def __init__(self, source=None):
        '''
        If header is provided, the geometry is read; if path to file is provided, the header is read from the file and then treated as such, or the is read from the file.

        :param source: From where the data should be acquiered/read?
        :type source: NoneType, or h5z.SaxspointH5, or path to h5z-file, or path to aares H5 file
        '''
        import aares.q_transformation as qt
        assert isinstance(source, h5z.SaxspointH5) or \
               isinstance(source, qt.ArrayQ) or \
               isinstance(source, str) or \
               source is None

        self._h5 = h5z.GroupH5(name='/')
        # self.geometry_fields = []
        self.attrs['aares_version'] = str(aares.version)
        self.attrs['aares_file_type'] = 'reduction_bins'

        if isinstance(source, str):
            if h5z.is_h5_file(source):
                if self.is_type(source):
                    self.read_from_file(source)
                else:
                    source = h5z.SaxspointH5(source)

        if isinstance(source, h5z.SaxspointH5):
            self.attrs['aares_version'] = str(aares.version)
            self.attrs['aares_detector_class'] = 'SaxspointH5'
            # self.read_geometry(source)
        if isinstance(source, qt.ArrayQ):
            self.attrs['aares_version'] = str(aares.version)
            self.attrs['aares_detector_class'] = 'SaxspointH5'

    @staticmethod
    def is_type(val):
        '''
        Check if the file is of the correct type.
        :param val:
        :return: bool
        '''
        import h5z, h5py
        out = []
        attributes = {}
        if h5z.is_h5_file(val):
            with h5z.FileH5Z(val, 'r') as fin:
                attributes.update(fin.attrs)

        else:
            raise TypeError('Input should be h5a file.')

        try:
            out.extend(['aares_detector_class' in attributes,
                        attributes['aares_file_type'] == 'reduction_bins'])

            if not attributes['aares_version'] == str(aares.version):
                logging.warning('AAres version used and of the file does not match: {}'.format(val))
        except KeyError:
            out.append(False)

        return all(out)

    def read_from_file(self, fin):
        '''
        Reads the data from a dedicated file
        :param fin:
        :return:
        '''
        self.read_header(fin)
        # self.geometry_fields = self['entry'].walk()
        if not self.attrs['aares_version'] == str(aares.version):
            logging.warning('AAres version used and of the file does not match.')

    def write_to_file(self, fout, mode='w'):
        '''
        Writes the array and geometry to the file.

        :param fout:
        :param mode:
        :return:
        '''

        self.write(fout, mode=mode, compression='gzip')

    def create_bins(self, arrQ, **kwargs):  # qmin=None, qmax=None, bins=None, frame_mask=None):
        '''
        Performs binning. Calculates integration masks and stores the masks

        :param arrQ: Q-transformed frame e.g. array, which holds q-value for each pixel
        :type arrQ: np.array
        :param qmin: q_min value
        :type qmin: float
        :param qmax: q_max value
        :type qmax: float
        :param bins: number of bins
        :type bins: int
        :param frame_mask: Additional mask for the frame.
        :type frame_mask: np.array(bools)
        :return: Returns array of q_values, and corresponding list of masks used for the integration
        '''

        qbins, bin_masks = prepare_bins(arrQ, **kwargs)
        self.q_axis = numpy.array(qbins)
        self.bin_masks = numpy.array(bin_masks)

    @property
    def bin_masks(self):
        '''
        Reduction bin masks
        :return:
        '''
        try:
            arrQ = self['/processing/reduction/bin_masks'][:]
        except KeyError:
            raise AttributeError('Bin maks were not calculated yet.')

        return arrQ

    @bin_masks.setter
    def bin_masks(self, val):
        data = h5z.DatasetH5(val, name='/processing/reduction/bin_masks')
        data.attrs['units'] = 'bool'
        data.attrs['long_name'] = 'Set of mask used for data reduction step.'
        self['/processing/reduction/bin_masks'] = data

    @property
    def q_axis(self):
        '''
        Naming of the bins on the q-axis
        :return:
        '''
        try:
            data = self['/processing/reduction/q_axis'][:]
        except KeyError:
            raise AttributeError('Q-vectors were not set yet.')

        return data

    @q_axis.setter
    def q_axis(self, val):
        dts = h5z.DatasetH5(val, name='/processing/reduction/q_axis')
        dts.attrs['units'] = 'nm^-1'
        dts.attrs['long_name'] = 'Description of q-axis.'
        self['/processing/reduction/q_axis'] = dts

    @property
    def number_bins(self):
        return len(self.bin_masks)

    @property
    def qmin(self):
        return min(self.q_axis)

    @property
    def qmax(self):
        return max(self.q_axis)


class JobReduction(aares.Job):

    def __set_meta__(self):
        super().__set_meta__()
        self._program_short_description = prog_short_description

    def __set_system_phil__(self):
        self.system_phil = phil_job_core

    def __help_epilog__(self):
        pass

    def __argument_processing__(self):
        pass

    def __process_unhandled__(self):
        for param in self.unhandled:
            if aares.datafiles.is_fls(param):
                self.params.input_files = param
            elif h5z.is_h5_file(param):
                if h5z.SaxspointH5.is_type(param):
                    self.params.input.file_name.append(param)
                elif aares.q_transformation.ArrayQ.is_type(param):
                    self.params.input.q_space = param
                elif ReductionBins.is_type(param):
                    self.params.reduction.file_bin_masks = param
                else:
                    raise aares.RuntimeErrorUser('Unknown type of H5 file: {}'.format(param))
            elif os.path.splitext(param)[1].lower() == '.png':
                self.params.input.mask = param
            else:
                raise aares.RuntimeErrorUser('Unknown input: {}'.format(param))

        # if len(self.unhandled) > 0:  # First file is input file
        #     if h5z.is_h5_file(self.unhandled[0]):
        #         self.params.input = self.unhandled[0]
        #     elif aares.datafiles.is_fls(self.unhandled[0]):
        #         self.params.input_files = self.unhandled[0]
        #     else:
        #         raise aares.RuntimeErrorUser('Unknown input: {}'.format(self.unhandled))
        #
        # if len(self.unhandled) == 2:  # Second file is output file
        #     root, ext = os.path.splitext(self.unhandled[1])
        #     if not 'h5a' in ext:
        #         raise aares.RuntimeErrorUser(
        #             'This should be output file in h5a-format: {}'.format(self.unhandled[1]))
        #     self.params.output = self.unhandled[1]
        # elif len(self.unhandled) > 2:
        #     raise aares.RuntimeErrorUser('Too many input parameters.')
        # else:
        #     pass

    def __worker__(self):

        if self.params.input_files is not None:
            imported_files = aares.datafiles.DataFilesCarrier(file_phil=self.params.input_files,
                                                              mainphil=phil_core)

            user_reduction_phil = phil_core.format(self.params)
            user_diff = phil_core.fetch_diff(user_reduction_phil)
            for group in imported_files.file_groups:
                aares.my_print('Preparing bins for {}...'.format(group.name))
                group_write = False
                if user_diff.as_str() != '':
                    group.update_group_phil(user_diff)
                    group_write = True

                if self.params.input.mask is not None:
                    group.mask = self.params.input.mask
                    group_write = True

                if group.mask is not None:  # TODO: maybe allow multiple?
                    aares.my_print('Reading mask...')
                    frame_mask = aares.mask.read_mask_from_image(group.mask, channel='A', invert=False)
                    px_used, px_total = aares.mask.count_used_pixels(frame_mask)
                    if px_used < 0.3* px_total:
                        logging.warning('Significant number of pixels will be ignored. Check your mask!')
                else:
                    frame_mask = None

                if group.group_phil.reduction.file_bin_masks is not None \
                        and os.path.isfile(group.group_phil.reduction.file_bin_masks) \
                        and (
                        not group_write):  # TODO: only if binnig parameters did not change.... No skipped when other like "beamstop" used
                    aares.my_print(
                        'Reading from file {}...'.format(group.group_phil.reduction.file_bin_masks))
                    group_bins = ReductionBins(group.group_phil.reduction.file_bin_masks)
                else:
                    aares.my_print('Reading q-space values...')
                    arrQ = aares.q_transformation.ArrayQ(group.q_space)
                    group_bins = ReductionBins(arrQ)
                    try:
                        group_bins.create_bins(arrQ.q_length,
                                           qmin=group.group_phil.reduction.q_range[0],
                                           qmax=group.group_phil.reduction.q_range[1],
                                           bins=group.group_phil.reduction.bins_number,
                                           frame_mask=frame_mask,
                                            skip_empty=self.params.reduction.empty_skip)
                    except AttributeError:
                        raise aares.RuntimeErrorUser('Q-space data expeceted, but do not exist. Please run aares.q-transformation.')

                    if group.group_phil.reduction.file_bin_masks is None:
                        group.group_phil.reduction.file_bin_masks = group.name + '.bins.h5a'
                        group_write = True

                    aares.my_print(
                        'Data were split into {bins} bins, spaning q-range from {qmin:.3f} nm-1 to {qmax:.2f} nm-1'.format(
                            bins=group_bins.number_bins,
                            qmin=group_bins.qmin,
                            qmax=group_bins.qmax
                        ))

                    aares.my_print('Binning mask for the group written to: {}'.format(
                        group.group_phil.reduction.file_bin_masks))
                    group_bins.write(group.group_phil.reduction.file_bin_masks)

                if group_write:
                    phil_out = group.write()
                    aares.my_print('Updated group work PHIL written to: {}'.format(phil_out))

            if self.params.output.input_files is None:
                self.params.output.input_files = 'binned.fls'
            imported_files.write_groups(file_out=self.params.output.input_files)

        else:
            raise aares.RuntimeWarningUser(
                'Not implemented yet, please use aares.import -> aares.q_transformation')

      #  aares.my_print('Using error model: {}\n'.format(self.params.reduction.error_model))
        for group in imported_files.file_groups:
            aares.my_print('Reducing files in group {}:'.format(group.name))

            integrate_group(group, imported_files.files_dict,
                            job_control=None,
                            export=self.params.export,
                            reduction=self.params.reduction,
                            output=self.params.output)  # TODO: prepare job_control

    # def __worker__(self):
    #
    #     if (((self.params.input is not None) and (self.params.input_files is not None)) or
    #             ((self.params.input is None) and (self.params.input_files is None))):
    #         raise aares.RuntimeErrorUser(
    #             'Exactly one of the parameters has to be set:\n\tinput\n\tinput_files')
    #
    #     if (self.params.input_files is not None) and (self.params.output is not None):
    #         logging.warning(
    #             'Output keyword is ignored, definitions from {} are used instead.'.format(
    #                 self.params.input_files))
    #
    #     to_process = []
    #     if self.params.input_files is not None:
    #         imported_files = aares.datafiles.DataFilesCarrier(file_phil=self.params.input_files,
    #                                                           mainphil=phil_core)
    #         for group in imported_files.file_groups:
    #             if group.q_space is None:
    #                 group.q_space = group.name + '.q_space.h5a'
    #
    #             to_process.append(
    #                 (group.q_space, ArrayQ(imported_files.files_dict[group.geometry])))
    #     elif self.params.input is not None:
    #         if self.params.output is None:
    #             self.params.output = self.params.input + '.q_space.h5a'
    #         aares.my_print('Reading file header...')
    #         to_process.append((self.params.output, ArrayQ(self.params.input)))
    #     else:
    #         raise AssertionError
    #
    #     aares.my_print('Performing Q-transformation')
    #     for fout, arrQ in to_process:
    #         arrQ.calculate_q()
    #         aares.my_print('Writing: {}'.format(fout))
    #         arrQ.write_to_file(fout)


def test():
    import aares.q_transformation as qt
    import time
    import aares.mask as am
    import aares.mask as mask

    fin = '../data/10x60s_826mm_010Frames.h5'

    h5hd = h5z.SaxspointH5(fin)

    frame_mask = numpy.logical_and(
        am.read_mask_from_image('frame_alpha_mask.png', channel='A', invert=True),
        am.detector_chip_mask(det_type='Eiger R 1M'))

    # frame_mask = am.read_mask_from_image('frame_alpha_mask.png',channel='A',invert=True)

    t0 = time.time()
    arrQ = qt.transform_detector_radial_q(h5hd)
    print(time.time() - t0)
    q_bins = create_bins(arrQ.min(), arrQ.max(), 750)
    q_vals = qt.get_q_axis(q_bins)
    print(time.time() - t0)
    q_masks = list_integration_masks(q_bins, arrQ, frame_mask=frame_mask)
    print(time.time() - t0)

    bincls = ReductionBins(h5hd)
    bincls.create_bins(arrQ)
    bincls.write_to_file('reduction.bins.h5')

    bincls2 = ReductionBins('reduction.bins.h5')

    with h5z.FileH5Z(fin) as h5f:
        frames = h5f['entry/data/data'][:]
        avr, std, num = integrate_mp(frames, bincls2.bin_masks)
    print(time.time() - t0)

    with open('data_826.dat', 'w') as fout:
        for q, I, s in zip(q_vals, avr, std):
            fout.write('{},{},{}\n'.format(q, I, s))

    import aares.statistics as stats
    import aares.draw2d as draw2d

    #  q_averages = stats.averages_to_frame_bins(q_masks, avr)
    #  q_stdevs = stats.averages_to_frame_bins(q_masks, std)
    #  draw2d.draw(q_averages,'frame_averages.png',Imax=1)

    # relative_dev_pix = stats.local_relative_deviation(frames[0], q_averages, window=15)
    # draw2d.draw(relative_dev_pix,'pix_dev.png', Imax=3, Imin=-3, cmap='PiYG')

    cc12 = []
    bin_sz = 5
    for chnk in pwr.chunks(q_masks, bin_sz):
        cc12.append(stats.set_cc12(frames[0], chnk))

    with open('cc12.dat', 'w') as fout:
        for i, cc in zip(range(int(750 / bin_sz)), cc12):
            fout.write('{} {}\n'.format(q_vals[bin_sz * i], cc))

    #   mask.draw_mask(q_masks[10],'q_mask10.png')

    with h5z.FileH5Z(fin) as fid:
        pass


def main():
    # test()
    job = JobReduction()
    return job.job_exit


if __name__ == '__main__':
    import sys

    sys.exit(main())
