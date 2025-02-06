import pytest

from fairchem.core.common.relaxation import ml_relaxation as mlr
from unittest.mock import patch, MagicMock

@patch.object(mlr.Batch, "from_data_list")
@patch.object(mlr, "data_list_collater")
@patch.object(mlr, "LBFGS")
@patch.object(mlr, "OptimizableUnitCellBatch")
@patch.object(mlr, "OptimizableBatch")
def test_ml_relax(obatch_mock, ounit_mock, lbfgs_mock, collater_mock, data_list_mock):
    batch = MagicMock()
    model = MagicMock()
    steps = 10
    fmax = 0.01
    batch.clone.return_value = 'clone'
    batch.sid = [1,2,3]
    relax_volume = False
    optimizer = MagicMock()
    lbfgs_mock.return_value = optimizer
    ounit_opt = MagicMock()
    ounit_opt.batch = ['item1', 'item2', 'item3']
    ounit_mock.return_value = ounit_opt
    obatch_mock.return_value = ounit_opt
    
    def data_list_patch(x):
        data = MagicMock()
        data.contents = x
        return data
    data_list_mock.side_effect = lambda x: data_list_patch(x)

    #1
    relax_cell = True
    result = mlr.ml_relax(batch, model, steps, fmax, relax_cell=relax_cell, relax_volume=relax_volume)
    ounit_mock.assert_called_with('clone', trainer=model, transform=None, mask_converged=True, hydrostatic_strain=relax_volume)
    lbfgs_mock.assert_called_with(optimizable_batch=ounit_opt, save_full_traj = True, traj_names=[1,2,3], traj_dir=None)
    optimizer.run.assert_called_with(fmax=fmax, steps=steps)
    assert result.contents == [ounit_opt.batch]

    #2
    ounit_mock.reset_mock()
    relax_cell = False
    result = mlr.ml_relax(batch, model, steps, fmax, relax_cell=relax_cell, relax_volume=relax_volume)
    ounit_mock.assert_not_called()
    obatch_mock.assert_called_with('clone', trainer=model, transform=None, mask_converged=True)
