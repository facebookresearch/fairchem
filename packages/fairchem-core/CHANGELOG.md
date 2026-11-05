<!--
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-->

# fairchem-core changelog

## Unreleased

- Fix `DFTD3Calculator` property export through ASE `Atoms.get_properties()`
  and calculation recipe runners. Component values previously stored in
  `calc.results["energy_contributions"]` (and the corresponding forces, stress,
  and free energy keys) are now tuples in `calc.contributions["energy"]`
  (or `"forces"`, `"stress"`, `"free_energy"`). Contributions are refreshed on
  every calculation; `calc.results` contains only summed ASE properties.
