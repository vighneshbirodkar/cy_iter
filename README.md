Cython Numpy Array Iterator
---------------------------

The purpose of these classes is to provide an easy-to-use thin interface around the NumPy C iterator API. To execute run

```shell
$ python setup.py build_ext --inplace
$ python test.py
```

Currently, the Cython code is about `4` times slower, likely because the sum function accesses the underlying array linearly.



## To Do
1. Implementing the [neighboorhood iterator](http://docs.scipy.org/doc/numpy/reference/c-api.array.html#neighborhood-iterator). This could enable writing ND code which is a lot faster than its Python `np.nditer` counterpart.

2. Replace `get_*` function with [Fused Types](http://docs.cython.org/src/userguide/fusedtypes.html). When I tired, fused types only seemed to work when the argument of the function was of a fixed type.
