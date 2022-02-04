import local_linear 
import math
import pytest
import os.path

X_raw = local_linear.file_import("xin.dms")
Y_raw = local_linear.file_import("yin.dms")

def is_float(data):
    for data_val in data:
        assert isinstance(data_val, float), "the input {0} is not a float".format(data_val)

def test_import_float():
    is_float(X_raw)
    is_float(Y_raw)
    
def test_import_lenght():
    assert len(X_raw) == len (Y_raw), "data length are not equal for xin and yin"

def test_get_weight():
    assert local_linear.get_weight(1.5, 1.4, 0.2) == math.exp(-((1.5 - 1.4) ** 2) / 0.2)
    assert local_linear.get_weight(1, 1, 0.5) == 1
    assert local_linear.get_weight(0.5, -0.5, 1) == math.exp(-1)
    assert local_linear.get_weight(0.4, -0.5, 0.2) == math.exp(-4.05)
    with pytest.raises(ZeroDivisionError):
        local_linear.get_weight(1.76, 1.75, 0)

def test_create_output():
    output_file = local_linear.main("xin.dms","yin.dms","output.dms",3,True)
    output_raw = local_linear.file_import(output_file)
    assert len(output_raw) == len(X_raw) == len(Y_raw)
    is_float(output_raw)