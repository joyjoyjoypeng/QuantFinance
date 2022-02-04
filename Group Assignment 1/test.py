'''
This is a test file for local_linear.py
'''
import math
from sqlalchemy import true
import local_linear
import pytest
X_raw = local_linear.file_import("xin.dms")
Y_raw = local_linear.file_import("yin.dms")
Output_file = local_linear.main("xin.dms","yin.dms","output.dms",3)

def test_main_arguments():
    '''
    This function tests whether invalid arguments raise an error
    '''
    with pytest.raises(TypeError):
        local_linear.main("xin.dms","yin.dms","output.dms")
    with pytest.raises(TypeError):
        local_linear.main("xin.dms","yin.dms","output.dms", 3, true, "xout.dms", "additional")
    with pytest.raises(ValueError):
        local_linear.main("xin.dms","yin.dms","output.dms","3")

def is_float(data):
    '''
    This function is used to test whether all values in a data set are floats
    '''
    for data_val in data:
        assert isinstance(data_val, float), "the input {0} is not a float".format(data_val)

def test_import_float():
    '''
    This function is used to test whether all values in the imported data set are floats
    '''
    is_float(X_raw)
    is_float(Y_raw)

def test_import_lenght():
    '''
    This function is used to test whether the imported data sets have matching length
    '''
    assert len(X_raw) == len (Y_raw), "data length are not equal for xin and yin"

def test_get_weight():
    '''
    This function is used to test whether the get_weight() function operates as expected
    '''
    assert local_linear.get_weight(1.5, 1.4, 0.2) == math.exp(-((1.5 - 1.4) ** 2) / 0.2)
    assert local_linear.get_weight(1, 1, 0.5) == 1
    assert local_linear.get_weight(0.5, -0.5, 1) == math.exp(-1)
    assert local_linear.get_weight(0.4, -0.5, 0.2) == math.exp(-4.05)
    with pytest.raises(ZeroDivisionError):
        local_linear.get_weight(1.76, 1.75, 0)

def test_main_output():
    '''
    This function is used to test whether the output data set is consisted of floats
    and has a matching length as the input data sets.
    '''
    output_raw = local_linear.file_import(Output_file)
    is_float(output_raw)
    assert len(output_raw) == len(X_raw) == len(Y_raw)
