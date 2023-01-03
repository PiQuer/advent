import pytest
import numpy as np
from numpy.lib import recfunctions as rfn

from utils import dataset_parametrization, DataSetBase


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


class DataSet(DataSetBase):
    def get_data(self):
        coordinates = rfn.structured_to_unstructured(
            np.fromregex(self.input_file, r'(\d+),(\d+)', dtype=np.dtype("int32,int32")))
        foldings = np.fromregex(self.input_file, r'fold along (x|y)=(\d+)', dtype=[('axis', 'S1'), ('num', int)])
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


round_1 = dataset_parametrization("2021", "13", [("", 17)], result=751, dataset_class=DataSet)
round_2 = dataset_parametrization("2021", "13", [("", expected_code_example)], result=expected_code,
                                  dataset_class=DataSet)


@pytest.mark.parametrize(**round_1)
def test_part_one(dataset: DataSet):
    data, foldings = dataset.get_data()
    assert fold(data, *tuple(foldings[0])).sum() == dataset.result


@pytest.mark.parametrize(**round_2)
def test_part_two(dataset: DataSet):
    data, foldings = dataset.get_data()
    for instruction in foldings:
        data = fold(data, *tuple(instruction))
    render = np.full(data.shape, ' ', dtype="<U1")
    render[data] = '#'
    assert '\n'.join([''.join(row) for row in render.transpose()]) == dataset.result
