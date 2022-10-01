import ipdb

def test_func():
    a = 1
    b = a
    return a+b


test_func()
ipdb.set_trace()
test_func()