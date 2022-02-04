def test_inputs_length ():
    x_raw = read_input("xin.dms")
    y_raw = read_input("yin.dms")
    assert len(x_raw) == len (y_raw), "data length are not equal for xin and yin"

def test_input_float ():
    x_raw = read_input("xin.dms")
    for x_vals in x_raw:
        x_float = float(x_vals)
        assert isinstance(x_float, float), "the x input {0} is not a float".format(x_vals)
    y_raw = read_input("yin.dms")
    for y_vals in y_raw:
        y_float = float(y_vals)
        assert isinstance(y_float, float), "the y input {0} is not a float".format(y_vals)

def read_input(file_name):
    with open("{0}".format(file_name),'r') as file:
        data = file.readlines()
    return data

if __name__ == '__main__':
    import sys
    sys.argv += ['--x','xin.dms','--y','yin.dms','--output','output.dms','--num_folds','10']
    from local_linear import *  
