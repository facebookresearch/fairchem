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

// Defines the operators.
//
// Functional schemas: each op allocates its output internally and
// returns a Tensor. This avoids the out-parameter aliasing issue that
// the old (Tensor C, ...) -> () signatures had — without an explicit
// `Tensor(a!) C` annotation, dynamo couldn't see that C was being
// mutated, so torch.compile-traced forwards would return uninitialized
// memory. Returning the output directly is functional, requires no
// aliasing annotation, and composes cleanly with both eager autograd
// and torch.compile + AOTAutograd.
TORCH_LIBRARY(fairchem_cpp, m) {
  m.def("segment_mm(Tensor A, Tensor B, Tensor seglen, bool b_trans) -> Tensor");
  m.def("segment_mm_backward(Tensor A, Tensor dC, Tensor seglen) -> Tensor");
}

