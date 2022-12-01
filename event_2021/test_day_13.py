import pytest
import numpy as np
from numpy.lib import recfunctions as rfn


expected_code_example = """
#####
#   #
#   #
#   #
#####
     
     
"""[1:-1]

expected_code = """
###   ##  #  # ###  #  # #    #  # #    
#  # #  # #  # #  # # #  #    # #  #    
#  # #    #### #  # ##   #    ##   #    
###  # ## #  # ###  # #  #    # #  #    
#    #  # #  # # #  # #  #    # #  #    
#     ### #  # #  # #  # #### #  # #### 
"""[1:-1]


def get_data(input_file):
    coordinates = rfn.structured_to_unstructured(
        np.fromregex(input_file, r'(\d+),(\d+)', dtype=np.dtype("int32,int32")))
    foldings = np.fromregex(input_file, r'fold along (x|y)=(\d+)', dtype=[('axis', 'S1'), ('num', int)])
    data = np.zeros(np.amax(coordinates, axis=0) + 1, dtype=bool)
    data[coordinates[:, 0], coordinates[:, 1]] = True
    return data, foldings


def fold(data, axis, num):
    result = data if axis == b'x' else data.transpose()
    # make sure we are always folding in the middle, otherwise this method does not work
    assert num == (result.shape[0]-1)/2
    index_left, index_right = slice(0, num), slice(-1, -1-num, -1)
    result = np.logical_or(result[(index_left, slice(None))], result[(index_right, slice(None))])
    return result if axis == b'x' else result.transpose()


@pytest.mark.parametrize("input_file,expected",
                         (("input/day_13_example.txt", 17),
                          ("input/day_13.txt", 751)))
def test_part_one(input_file, expected):
    data, foldings = get_data(input_file)
    result = fold(data, *tuple(foldings[0])).sum()
    assert expected == result


@pytest.mark.parametrize("input_file, expected",
                         (pytest.param("input/day_13_example.txt", expected_code_example, id="example"),
                          pytest.param("input/day_13.txt", expected_code, id="real")))
def test_part_two(input_file, expected):
    data, foldings = get_data(input_file)
    for instruction in foldings:
        data = fold(data, *tuple(instruction))
    render = np.full(data.shape, ' ', dtype="<U1")
    render[data] = '#'
    result = '\n'.join([''.join(row) for row in render.transpose()])
    assert result == expected
