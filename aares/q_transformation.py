import h5z
import numpy
import math
import copy
import aares.power as pwr
import aares
import aares.datafiles
import logging
import freephil as phil
import os

prog_short_description = 'Transforms file Q-space coordinates'

phil_core_str = '''
q_transformation {
units = *nm A
.type = choice
.help = 'Units to be used. The length of scattering vector (:math:`q`) is calculated as :math:`4\\pi*sin \\theta/\\lambda` in inverse units of choice.'

geometry {
    beam_center_px = None
    .type = floats(2)
    .help = Beam position at the detector in pixels

    sdd = None
    .type = float
    .help = Sample to detector distance in meters
    
    meridional_angle = None
    .type = float
    .help = Meridional angle in radians
    
    pixel_size = None
    .type = floats(2)
    .help = Pixel size (X,Y) in meters
    
    wavelength = None
    .type = float
    .help = Wavelength in meters 
}    

file = None
.type = path
.help = File storing q-space coordinates. Aares H5 file.
}
'''

phil_core = phil.parse(phil_core_str)

phil_prog = phil.parse('''
include scope aares.common.phil_input_files

input = None
.type = path
.help = Input file to be processed. Single H5-like file is allowed, for multiple files, use AAres importing protocol

output = None
.type = path
.help = Output file. Used only with single file input.

    ''' + phil_core_str, process_includes=True
                       )


def get_item_and_type(data):
    '''
    Extracts a value from single-item dataset, and type as string
    :param data:
    :return: value, type
    '''

    dat_item = data.item()
    if data.dtype.kind in ['U', 'S']:
        value = dat_item.decode()
        val_type = 'str'
    else:
        value = dat_item
        val_type = type(value).__name__

    return value, val_type


def dataset_to_scope(dataset, name=None, **kwargs):
    '''
    Converts DatasetH5 into PHIL scope
    :param dataset:
    :return:
    '''

    if name is None:
        name = dataset.name

    if dataset.size > 1:
        value = 'long array, skipped'
    else:
        value, val_type = get_item_and_type(dataset)

    value_definition = phil.definition('value',
                                       ["{}".format(value)],
                                       type=val_type,
                                       help='Parameter value')
    attrs_objects = []
    for key, attr in dataset.attrs.items():
        val, val_type = get_item_and_type(attr)
        attrs_objects.append(phil.definition(key, [val], type=val_type))

    attributes_scope = phil.scope('attrs',
                                  objects=attrs_objects,
                                  help='Parameter attributes')

    return phil.scope(name,
                      objects=[value_definition, attributes_scope],
                      **kwargs)


def extract_geometry_to_phil(header, name='detector', parent='/entry/instrument/detector',
                             skip=['time', 'x_pixel_offset', 'y_pixel_offset', 'pixel_mask',
                                   'geometry']):
    '''
    Extract geometry information from the header to a PHIL scope
    :param header:
    :param parent:
    :return:
    '''

    detector = header[parent]

    objects = []

    for key, val in detector.items():
        if key in skip:
            continue
        if isinstance(val, h5z.DatasetH5):
            objects.append(dataset_to_scope(val, key))

    return phil.scope(name,
                      objects)


def rotate_by_axis_matrix(axis, angle):
    '''

    :param axis: vector of rotation axis
    :param angle: rotation angle
    :return: rotation matrix (numpy.array 3x3)
    '''
    size_axis = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    ux = axis[0] / size_axis
    uy = axis[1] / size_axis
    uz = axis[2] / size_axis

    cT = math.cos(angle)
    sT = math.sin(angle)

    return numpy.array(
        [[cT + ux * ux * (1 - cT), ux * uy * (1 - cT) - uz * sT, ux * uz * (1 - cT) + uy * sT],
         [uy * ux * (1 - cT) + uz * sT, cT + uy * uy * (1 - cT), uy * uz * (1 - cT) - ux * sT],
         [uz * ux * (1 - cT) - uy * sT, uz * uy * (1 - cT) + ux * sT, cT + uz * uz * (1 - cT)]])


def translate_by_vector(vector, distance):
    '''

    :param vector: direction vector (is unified)
    :param distance: distance of the translation
    :return:
    '''
    size_vec = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    ux = vector[0] / size_vec
    uy = vector[1] / size_vec
    uz = vector[2] / size_vec

    return distance * numpy.array([ux, uy, uz])


def vector_from_string(vecin):
    '''
    Converts vector from comma separeted string to numpy array
    :param vecin:
    :return: numpy.arrat
    '''

    ls = vecin.split(',')
    return numpy.array(ls, dtype=float)


def rotation_factory(rotation_matrix):
    def operator(x):
        return rotation_matrix @ x

    return operator


def translation_factory(translation_vector):
    def operator(x):
        return x + translation_vector

    return operator


def detector_real_space(header):
    '''
    Transforms detector pixels into the realspace coordinates
    :param header: Header or open H5 file
    :return: X, Y, Z numpy.arrays of coordinates
    '''

    x0 = header['entry/instrument/detector/x_pixel_offset'][:]
    y0 = header['entry/instrument/detector/y_pixel_offset'][:]
    # x0 = numpy.array([1,2,3])
    # y0 = numpy.array([6,7,8,9])
    zero = numpy.array([0])

    X0, Y0 = numpy.meshgrid(y0, x0)
    # Z0 = numpy.zeros(X0.shape)

    XYZ = numpy.array(numpy.meshgrid(x0, y0, zero)).T.reshape(-1, 3)

    detector = 'entry/instrument/detector'
    transformation = header[detector]
    transformation.attrs['depends_on'] = transformation['depends_on'].item()

    while 'depends_on' in transformation.attrs:
        transformation = header[detector + '/' + transformation.attrs['depends_on'].decode()]
        if transformation.attrs['transformation'].decode() == 'translation':
            u_vec = vector_from_string(transformation.attrs['vector'].decode())
            translation_vector = translate_by_vector(u_vec, transformation.item())
            XYZ = XYZ + translation_vector
        elif transformation.attrs['transformation'].decode() == 'rotation':
            u_vec = vector_from_string(transformation.attrs['vector'].decode())
            rotation_matrix_T = rotate_by_axis_matrix(transformation.attrs['vector'],
                                                      transformation.item()).T
            XYZ = XYZ @ rotation_matrix_T
        else:
            raise KeyError('Wrong transformation type:' + transformation['transformation'])

    return XYZ


def get_q_axis(bins):
    '''
    Return q-values as X-axis labels
    :param bins:
    :return: np.array
    '''
    l = [(b[1] - b[0]) / 2 + b[0] for b in bins]
    return numpy.array(l)


def get_q(XYZ, beam_vec, wavelength=1):
    '''
    Calculates radial q-values for the pixels
    :param XYZ: array of vectors ( dimensions 3,N)
    :param beam_vec: beam vector
    :param wavelength: X-ray wavelength. Units define output units.
    :return: numpy.array of q-values, 1D
    '''
    beam_norm = math.sqrt(
        beam_vec[0] * beam_vec[0] + beam_vec[1] * beam_vec[1] + beam_vec[2] * beam_vec[
            2]) * beam_vec
    XYZT = XYZ.T
    cos2T = (beam_norm @ XYZT) / numpy.sqrt(
        XYZT[0] * XYZT[0] + XYZT[1] * XYZT[1] + XYZT[2] * XYZT[2])
    sinT = numpy.sqrt((1 - cos2T) / 2)
    return 4 * math.pi * sinT / wavelength


def transform_detector_radial_q(header, beam=(0, 0, 1), unit='nm'):
    '''
    Perform q-transformation of the detector based on the header.
    Length of q-vector is returned for each pixel in an array corresponding to the frame.

    :param header: SaxspointH5 File header
    :return: numpy.array of frame dimensions
    '''

    if unit == 'nm':
        unit_multiplier = 1e-9
    elif unit == 'A':
        unit_multiplier = 1e-10
    else:
        raise AttributeError('Unknown unit: {}'.format(unit))

    if not ((wl_unit := header['entry/instrument/monochromator/wavelength'].attrs[
        'units'].decode()) == 'm'):
        raise aares.RuntimeErrorUser('Unknown wavelength unit on input: {}'.format(
            wl_unit))

    if beam is not numpy.ndarray:
        beam = numpy.array(beam)

    XYZ = detector_real_space(header)

    Q = get_q(XYZ, beam, header.wavelength / unit_multiplier)
    x0 = header['entry/instrument/detector/x_pixel_offset'][:]
    y0 = header['entry/instrument/detector/y_pixel_offset'][:]
    arrQ = Q.reshape([len(x0), len(y0)], order='C').T

    return numpy.ascontiguousarray(arrQ)


class ArrayQ(h5z.SaxspointH5):
    '''
    Detector surface transformed to Q-values
    '''

    # TODO: transform to Class factory

    skip_entries = []

    def __init__(self, source=None):
        '''
        If header is provided, the geometry is read; if path to file is provided, the header is read from the file and then treated as such, or the is read from the file.

        :param source: From where the data should be acquiered/read?
        :type source: NoneType, or h5z.SaxspointH5, or path to h5z-file, or path to aares H5 file
        '''

        assert isinstance(source, h5z.SaxspointH5) or \
               isinstance(source, str) or \
               source is None

        self._h5 = h5z.GroupH5(name='/')
        self.geometry_fields = [] #TODO: should not be here? inherited from the class...
        self.attrs['aares_version'] = str(aares.version)
        self.attrs['aares_file_type'] = 'q_space'

        if isinstance(source, str):
            if h5z.is_h5_file(source):
                if self.is_type(source):
                    self.read_from_file(source)
                else:
                    source = h5z.SaxspointH5(source)

        if isinstance(source, h5z.SaxspointH5):
            self.attrs['aares_version'] = str(aares.version)
            self.attrs['aares_detector_class'] = 'SaxspointH5'
            self.read_geometry(source)

    def read_geometry(self, source):
        '''
        Reads geometry fields from the source header
        :param source: Header object to be read from
        :type source: h5z.SaxspointH5
        :return:
        '''

        assert isinstance(source, h5z.SaxspointH5)

        self.geometry_fields = copy.copy(source.geometry_fields)

        for key in self.geometry_fields:
            self[key] = copy.deepcopy(source[key])

    def modify_geometry(self, geometry): #*args, **kwargs):
        '''
        Modify geometry parameters, which are available as class members
        '''

        # for arg in args:
        #     if isinstance(arg, phil.scope_extract):
        #         kwargs.update(arg.__dict__)
        # try:
        #     for k, v in geometry.__dict__.items():
        #         if v is not None:
        #             self.__dict__[k] = v
        #             logging.debug('Geometry parameter updated: {}'.format(k))
        # except KeyError:
        #     raise AttributeError('Unknown geometry parameter: {}'.format(k))
        # pass
        pass

        if geometry.beam_center_px is not None:
            self.beam_center_px = geometry.beam_center_px
        if geometry.meridional_angle is not None:
            self.meridional_angle = geometry.meridional_angle
        if geometry.pixel_size is not None:
            self.pixel_size = geometry.pixel_size
        if geometry.wavelength is not None:
            self.wavelength = geometry.wavelength
        if geometry.sdd is not None:
            self.sdd = geometry.sdd

    def read_from_file(self, fin):
        '''
        Reads the data from a dedicated file
        :param fin:
        :return:
        '''
        self.read_header(fin)
        self.geometry_fields = self['entry'].walk()
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

    def calculate_q(self, beam=(0, 0, 1), unit='nm'):
        '''
        Performs the Q-transformation
        :return:
        '''
        self.q_length = transform_detector_radial_q(self, beam=beam, unit=unit)

    @property
    def geometry_fields(self):
        try:
            val = self._geometry_fields
        except AttributeError:
            val = []
            self._geometry_fields = val
        return val

    @geometry_fields.setter
    def geometry_fields(self, val):
        self._geometry_fields = val

    @property
    def q_length(self):
        '''
        Returns an array of q-vector lenghts for individual pixels
        :return:
        '''
        try:
            arrQ = self['/processing/q_vector/length'][:]
        except KeyError:
            raise AttributeError('Q-vectors were not set yet.')

        return arrQ

    @q_length.setter
    def q_length(self, val):
        out_q = h5z.DatasetH5(val, name='/processing/q_vector/length')
        out_q.attrs['units'] = 'nm^-1'
        out_q.attrs['long_name'] = 'Length of the q-vector corresponding to given pixel'
        self['/processing/q_vector/length'] = out_q

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
                # if all([fin.get(it, default=False) == True for it in['/processing/q-vector/length']]):
                #     out.append(True)
                # else:
                #     out.append(False)
        #        elif isinstance(val, h5z.GroupH5) or isinstance(val,h5py.Group):
        #            attributes.update(val.attrs)
        else:
            raise TypeError('Input should be h5a file.')

        try:
            out.extend(['aares_detector_class' in attributes,
                        attributes['aares_file_type'] == 'q_space'])

            if not attributes['aares_version'] == str(aares.version):
                logging.warning('AAres version used and of the file does not match.')
        except KeyError:
            out.append(False)

        return all(out)


class JobQtrasform(aares.Job):

    def __set_meta__(self):
        super().__set_meta__()
        self._program_short_description = prog_short_description

    def __set_system_phil__(self):
        self.system_phil = phil_prog

    def __help_epilog__(self):
        pass

    def __argument_processing__(self):
        pass

    def __process_unhandled__(self):
        if len(self.unhandled) > 0:  # First file is input file
            if h5z.is_h5_file(self.unhandled[0]):
                self.params.input = self.unhandled[0]
            elif aares.datafiles.is_fls(self.unhandled[0]):
                self.params.input_files = self.unhandled[0]
            else:
                raise aares.RuntimeErrorUser('Unknown input: {}'.format(self.unhandled))

        if len(self.unhandled) == 2:  # Second file is output file
            root, ext = os.path.splitext(self.unhandled[1])
            if not 'h5a' in ext:
                raise aares.RuntimeErrorUser(
                    'This should be output file in h5a-format: {}'.format(self.unhandled[1]))
            self.params.output = self.unhandled[1]
        elif len(self.unhandled) > 2:
            raise aares.RuntimeErrorUser('Too many input parameters.')
        else:
            pass

    def __worker__(self):

        if (((self.params.input is not None) and (self.params.input_files is not None)) or
                ((self.params.input is None) and (self.params.input_files is None))):
            raise aares.RuntimeErrorUser(
                'Exactly one of the parameters has to be set:\n\tinput\n\tinput_files')

        if (self.params.input_files is not None) and (self.params.output is not None):
            logging.warning(
                'Output keyword is ignored, definitions from {} are used instead.'.format(
                    self.params.input_files))
        in_geometry = self.params.q_transformation.geometry
        modify_geometry = any([in_geometry.beam_center_px is not None,
                               in_geometry.meridional_angle is not None,
                               in_geometry.pixel_size is not None,
                               in_geometry.sdd is not None,
                               in_geometry.wavelength is not None])
        to_process = []
        if self.params.input_files is not None:
            imported_files = aares.datafiles.DataFilesCarrier(file_phil=self.params.input_files,
                                                              mainphil=phil_core)
            for group in imported_files.file_groups:
                if group.q_space is None:
                    group.q_space = group.name + '.q_space.h5a'

                arrQ = ArrayQ(imported_files.files_dict[group.geometry])
                if modify_geometry:
                    group.group_phil.q_transformation.geometry = in_geometry
                if any([val is not None for val in group.group_phil.q_transformation.geometry.__dict__.values()]):
                    arrQ.modify_geometry(group.group_phil.q_transformation.geometry)
                to_process.append(
                    (group.q_space, arrQ))
            aares.my_print('Updating {}...'.format(self.params.input_files))
            imported_files.write_groups(self.params.input_files)
        elif self.params.input is not None:
            if self.params.output is None:
                self.params.output = self.params.input + '.q_space.h5a'
            aares.my_print('Reading file header...')
            arrQ = ArrayQ(self.params.input)
            if modify_geometry:
                arrQ.modify_geometry(in_geometry)
            to_process.append((self.params.output, arrQ))
        else:
            raise AssertionError

        aares.my_print('Performing Q-transformation')
        for fout, arrQ in to_process:

            arrQ.calculate_q()
            aares.my_print('Writing: {}'.format(fout))
            arrQ.write_to_file(fout)



def test(fin):
    import time
    import matplotlib.pyplot as plt

    h5in = h5z.SaxspointH5(fin)

    #   h5in.write('test_out.h5', skipped=True)

    cArrQ = ArrayQ(h5in)

    #   print(all(h5in[match] == cArrQ[match] for match in h5in.geometry_fields))

    # a = numpy.array([[1,2,3],
    #                  [4,5,6],
    #                  [7,8,9],
    #                  [1,2,3]])
    #
    # aq = get_q(a, numpy.array([1,0,0]))

    t0 = time.time()
    arrQ = transform_detector_radial_q(h5in, (0, 0, 1))
    dt1 = time.time() - t0

    print(dt1)

    cArrQ.q_length = arrQ

    cArrQ.write_to_file('q-r.h5')

    cArrQ2 = ArrayQ('q-r.h5')

    lng = cArrQ.q_length

    qmin = numpy.amin(arrQ)
    qmax = numpy.max(arrQ)
    idx_qmin = numpy.where(lng == qmin)

    print(qmin, qmax)

    print(arrQ[245, 526])
    print(arrQ[526, 246])

    with h5z.FileH5Z(fin) as h5f:
        fr = h5f['entry/data/data'][0]

    # plt.imshow(fr)
    plt.imshow(lng)
    plt.show()
    det_pos = numpy.array([0.1, .2, 0])

    #    t1 =time.time()
    #    trans_list = get_detector_transformation_list(h5in)
    #    det_trans = get_composition_operator(trans_list)
    #    pixel_XYZ = transform_detector(h5in,det_trans)
    #    dt2 = time.time() - t1
    #    print(dt1,dt2)
    pass


def main():
    #  test('../data/AgBeh_826mm.h5z')

    job = JobQtrasform()
    return job.job_exit


if __name__ == '__main__':
    import sys
    sys.exit(main())
