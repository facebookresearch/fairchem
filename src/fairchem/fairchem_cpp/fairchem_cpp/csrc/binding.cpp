#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  PyObject* PyInit__C(void) {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

// Defines the operators
TORCH_LIBRARY(fairchem_cpp, m) {
  m.def("segment_mm(Tensor A, Tensor B, Tensor C, Tensor seglen, bool b_trans) -> ()");
  m.def("segment_mm_backward(Tensor A, Tensor dC, Tensor dB, Tensor seglen) -> ()");
}

