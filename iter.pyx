# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
from cpython.ref cimport PyObject
import numpy as np
cimport numpy as cnp


cnp.import_array()


cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArrayIterObject:
        int index
        int size
        char* dataptr
        pass


cdef extern from "numpy/arrayobject.h":
    cdef PyObject* PyArray_IterNew(PyObject* arr)
    ctypedef struct PyArrayObject:
        pass
    cdef void PyArray_ITER_NEXT(PyArrayIterObject* iter)


cdef class cy_iter:
    cdef int len
    cdef PyArrayObject* array
    cdef PyArrayIterObject* iter;
    cdef PyObject* obj

    def __init__(self, arr):
        self.iter = <PyArrayIterObject*>PyArray_IterNew(<PyObject*>arr);

    cdef inline bint next(self):
        PyArray_ITER_NEXT(self.iter)
        return self.iter.index < self.iter.size

    cdef inline cnp.float_t get_float_t(self):
        cdef cnp.float_t *address = <cnp.float_t*>self.iter.dataptr
        return address[0]

    cdef inline cnp.int_t get_int_t(self):
        cdef cnp.int_t *address = <cnp.int_t*>self.iter.dataptr
        return address[0]


def run():
    import time
    arr = np.random.rand(800, 600)*10
    arr = arr.astype(np.int)

    t = time.time()
    ans = np.sum(arr)
    print "Numpy took ",time.time() - t
    print "ans = ", ans

    t = time.time()
    cdef cnp.long_t c_ans = 0
    cdef cy_iter iter = cy_iter(arr)

    c_ans += iter.get_int_t()

    while iter.next():
        c_ans += iter.get_int_t()

    print "Cython took ", time.time() - t
    print "ans = ",c_ans

    py_iter = np.nditer(arr)

    py_ans = 0
    t = time.time()
    for val in py_iter:
        py_ans += val

    print "Python took ", time.time() - t
    print "ans = ", py_ans
