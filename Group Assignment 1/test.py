'''
This is a test file for local_linear.py
'''
import math
import local_linear
import pytest
X_raw = local_linear.file_import("xin.dms")
Y_raw = local_linear.file_import("yin.dms")
Output_file = local_linear.main("xin.dms","yin.dms","output.dms",3)

def test_main_arguments():
    '''
    This function tests whether invalid arguments raise an error

    Inputs: 
        -- No inputs are required as this function assumes an error will be raised
        when insufficient arguments are passed to the main function -- 

    Returns:
        -- Nothing should be returned ideally if only the predicted errors are raised --
    '''
    with pytest.raises(TypeError):
        local_linear.main("xin.dms","yin.dms","output.dms")
    with pytest.raises(TypeError):
        local_linear.main("xin.dms","yin.dms","output.dms", 3, True, "xout.dms", "additional")
    with pytest.raises(ValueError):
        local_linear.main("xin.dms","yin.dms","output.dms","3")

def is_float(data):
    '''
    This function is used to test whether all values in a data set are floats
    
    Inputs: 
        data (array): An array of values that are to be tested

    Returns:
        -- A bool will be returned in the technical sense, "True" if all the values in the
        data set are indeed floats, "False" + an error statement if a particular value is
        flagged out to be not a float --
    '''
    for data_val in data:
        assert isinstance(data_val, float), "the input {0} is not a float".format(data_val)

def test_import_float():
    '''
    This function is used to test whether all values in the imported data set are floats
    
    Inputs: 
        -- No inputs are required as this function uses the arrays defined 
        earlier in the file --

    Returns:
        -- Nothing should be returned ideally if both arrays of x and y values
        only contain floats --
    '''
    is_float(X_raw)
    is_float(Y_raw)

def test_import_lenght():
    '''
    This function is used to test whether the imported data sets have matching length
    
    Inputs: 
        -- No inputs are required as this function uses the arrays defined 
        earlier in the file --

    Returns:
        -- Nothing should be returned ideally if both arrays of x and y values
        are of the same length --
    '''
    assert len(X_raw) == len (Y_raw), "data length are not equal for xin and yin"

def test_get_weight():
    '''
    This function is used to test whether the get_weight() function operates as expected
    
    Inputs: 
        -- No inputs are required as this function internally creates scenarios 
        to be tested --

    Returns:
        -- A bool will be returned in the technical sense, "True" if all the equalities happen
        to be true, indicating that the "get_weight()" function works exactly like it is
        supposed to, "False" if an equality doesn't match up or if expected error does not
        get raised --
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
    
    Inputs: 
        -- No inputs are required as this function internally creates scenarios
        to be tested --

    Returns:
        -- A bool will be returned in the technical sense, "True" if all the values in the
        final output are indeed floats and are of the same length as the original sets of
        x and y values, "False" if any of the above conditions are not met --
    '''
    output_raw = local_linear.file_import(Output_file)
    is_float(output_raw)
    assert len(output_raw) == len(X_raw) == len(Y_raw)
