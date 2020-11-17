import h5z
import h5py
import numpy
import math
import copy

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

def get_detector_transformation_list(header):
    '''
    Reads the header and creates list of operators (function objects) for each detector transformation
    :param header: Header or open H5 file
    :return: list of transformation operators as a function
    '''
    detector = 'entry/instrument/detector'
    transformation = header[detector]
    transformation.attrs['depends_on'] = transformation['depends_on'].item()
    operators_out = []
    while 'depends_on' in transformation.attrs:
        transformation = header[detector+'/' + transformation.attrs['depends_on'].decode()]
        if transformation.attrs['transformation'].decode() == 'translation':
            u_vec = vector_from_string(transformation.attrs['vector'].decode())
            translation_vector = translate_by_vector(u_vec, transformation.item())
            operator = translation_factory(translation_vector)
        elif transformation.attrs['transformation'].decode() == 'rotation':
            u_vec = vector_from_string(transformation.attrs['vector'].decode())
            rotation_matrix = rotate_by_axis_matrix(transformation.attrs['vector'],transformation.item())
            operator = rotation_factory(rotation_matrix)
        else:
            raise KeyError('Wrong transformation type:' + transformation['transformation'])
        operators_out.append(copy.deepcopy(operator))

    return operators_out

def get_composition_operator(operator_list):
    '''
    Creates a single function which applies list of operators at the input vector
    :param operator_list: list of functions
    :return: function
    '''

    def operator(vec_in):
        op_ls = operator_list
        vec_out = copy.deepcopy(vec_in)
        for op in op_ls:
            vec_out = op(vec_out)

        return vec_out

    return operator

def test(fin):
    h5in = h5z.SaxspointH5(fin)
    det_pos = numpy.array([0.1, .2, 0])
    trans_list = get_detector_transformation_list(h5in)
    pos_xyz = get_composition_operator(trans_list)(det_pos)

    pass

def main():
    test('data/AgBeh_826mm.h5z')



if __name__ == '__main__':
    main()