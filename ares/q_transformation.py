import h5z
import h5py
import numpy
import numexpr3 as numexpr
import math
import copy
import ares.power as pwr

def rotate_by_axis_matrix(axis, angle):
    '''

    :param axis: vector of rotation axis
    :param angle: rotation angle
    :return: rotation matrix (numpy.array 3x3)
    '''
    size_axis = math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    ux = axis[0] / size_axis
    uy = axis[1] / size_axis
    uz = axis[2] / size_axis

    cT = math.cos(angle)
    sT = math.sin(angle)

    return numpy.array([[cT + ux*ux*(1 - cT)   , ux*uy*(1 - cT)- uz*sT , ux*uz*(1 - cT) + uy*sT],
                        [uy*ux*(1 - cT) + uz*sT, cT + uy * uy *(1 - cT), uy*uz*(1 - cT) - ux*sT],
                        [uz*uz*(1 - cT) - uy*sT, uz*uy*(1 - cT) + ux*sT, cT + uz*uz*(1 - cT)   ]])

def translate_by_vector(vector, distance):
    '''

    :param vector: direction vector (is unified)
    :param distance: distance of the translation
    :return:
    '''
    size_vec = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
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
    #x0 = numpy.array([1,2,3])
    #y0 = numpy.array([6,7,8,9])
    zero = numpy.array([0])

    X0, Y0 = numpy.meshgrid(y0,x0)
   # Z0 = numpy.zeros(X0.shape)

    XYZ = numpy.array(numpy.meshgrid(x0,y0,zero)).T.reshape(-1,3)

    detector = 'entry/instrument/detector'
    transformation = header[detector]
    transformation.attrs['depends_on'] = transformation['depends_on'].item()

    while 'depends_on' in transformation.attrs:
        transformation = header[detector+'/' + transformation.attrs['depends_on'].decode()]
        if transformation.attrs['transformation'].decode() == 'translation':
            u_vec = vector_from_string(transformation.attrs['vector'].decode())
            translation_vector = translate_by_vector(u_vec, transformation.item())
            XYZ = XYZ + translation_vector
        elif transformation.attrs['transformation'].decode() == 'rotation':
            u_vec = vector_from_string(transformation.attrs['vector'].decode())
            rotation_matrix_T = rotate_by_axis_matrix(transformation.attrs['vector'],transformation.item()).T
            XYZ = XYZ @ rotation_matrix_T
        else:
            raise KeyError('Wrong transformation type:' + transformation['transformation'])

    return XYZ


def get_q(XYZ, beam_vec, wavelength = 1):
    '''
    Calculates radial q-values for the pixels
    :param XYZ: array of vectors ( dimensions 3,N)
    :param beam_vec: beam vector
    :param wavelength: X-ray wavelength. Units define output units.
    :return: numpy.array of q-values, 1D
    '''
    beam_norm = math.sqrt(beam_vec[0]*beam_vec[0]+beam_vec[1]*beam_vec[1]+beam_vec[2]*beam_vec[2]) * beam_vec
    XYZT = XYZ.T
    cos2T = (beam_norm @ XYZT) / numpy.sqrt(XYZT[0]*XYZT[0]+XYZT[1]*XYZT[1]+XYZT[2]*XYZT[2])
    sinT = numpy.sqrt((1-cos2T)/2)
    return sinT/wavelength

def transform_detector_radial_q(header, beam = (0,0,1), unit='nm'):
    '''
    Perform q-transformation of the detector based on the header. Size of q-vector is returned for each pixel in an array coresponding to the frame.
    :param header: SaxspointH5 File header
    :return: numpy.array of frame dimensions
    '''

    if unit == 'nm':
        unit_multiplier = 10e-9
    elif unit == 'A':
        unit_multiplier = 10e-10
    else:
        raise AttributeError('Unknown unit: {}'.format(unit))

    if not ((wl_unit := header['entry/instrument/monochromator/wavelength'].attrs['units'].decode()) == 'm'):
        raise IOError('Unknown wavelength unit on input: {}'.format(wl_unit)) # TODO: Change to user error, when logging and stuff in place


    if beam is not numpy.ndarray:
        beam = numpy.array(beam)

    XYZ = detector_real_space(header)

    Q = get_q(XYZ, beam, header.wavelength/unit_multiplier)
    x0 = header['entry/instrument/detector/x_pixel_offset'][:]
    y0 = header['entry/instrument/detector/y_pixel_offset'][:]
    arrQ = Q.reshape([len(x0),len(y0)], order='C').T

    return arrQ

def test(fin):

    import time
    import matplotlib.pyplot as plt

    h5in = h5z.SaxspointH5(fin)

    a = numpy.array([[1,2,3],
                     [4,5,6],
                     [7,8,9],
                     [1,2,3]])

    aq = get_q(a, numpy.array([1,0,0]))

    t0 = time.time()
    arrQ = transform_detector_radial_q(h5in, (0,0,1))
    dt1 = time.time() - t0

    print(dt1)


    qmin = numpy.amin(arrQ)
    qmax = numpy.max(arrQ)
    idx_qmin = numpy.where(arrQ == qmin)

    print(qmin, qmax)

    print(arrQ[245, 526])
    print(arrQ[245, 513])

    with h5z.FileH5Z(fin) as h5f:
        fr = h5f['entry/data/data'][0]

    #plt.imshow(fr)
    plt.imshow(arrQ)
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
    test('data/AgBeh_826mm.h5z')



if __name__ == '__main__':
    main()