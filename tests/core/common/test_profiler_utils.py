"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from fairchem.core.common.profiler_utils import (
    RecordFunctionWithBackward,
    mark_backward,
    record_backward,
    record_function_with_backward,
)


class TestRecordBackwardDecorator:
    """Tests for the @record_backward decorator (automatic marking)."""

    def test_forward_and_backward_labels_appear(self):
        """Test that both forward and backward labels appear in trace."""

        @record_backward("my_func")
        def my_func(x):
            return x * 2 + 1

        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            y = my_func(x)
            y.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "my_func" in event_names
        assert "my_func_backward" in event_names

    def test_tuple_output(self):
        """Test that tuple outputs are all marked."""

        @record_backward("multi_output")
        def multi_output(x):
            return x * 2, x + 1

        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            y1, y2 = multi_output(x)
            (y1.sum() + y2.sum()).backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "multi_output" in event_names
        assert "multi_output_backward" in event_names

    def test_gradient_correctness(self):
        """Test gradients are correct through decorated function."""

        @record_backward("square")
        def square(x):
            return x**2

        x = torch.randn(10, 10, requires_grad=True)
        y = square(x)
        y.sum().backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, 2 * x)


class TestRecordFunctionWithBackward:
    """Tests for the record_function_with_backward context manager."""

    def test_forward_label_appears_in_trace(self):
        """Test that the forward pass label appears in the profiler trace."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("test_block") as ctx:
                y = x * 2
                y = ctx.mark(y)
            y.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "test_block" in event_names

    def test_backward_label_appears_in_trace(self):
        """Test that the backward pass label appears in the profiler trace."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("test_block") as ctx:
                y = x * 2
                y = ctx.mark(y)
            y.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "test_block_backward" in event_names

    def test_gradient_flows_correctly(self):
        """Test that gradients flow correctly through the marked tensor."""
        x = torch.randn(10, 10, requires_grad=True)

        with record_function_with_backward("test_block") as ctx:
            y = x * 2
            y = ctx.mark(y)

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full_like(x.grad, 2.0))

    def test_no_grad_tensor_not_marked(self):
        """Test that tensors without requires_grad are returned unchanged."""
        x = torch.randn(10, 10, requires_grad=False)

        with record_function_with_backward("test_block") as ctx:
            y = x * 2
            result = ctx.mark(y)

        assert result is y

    def test_nested_context_managers(self):
        """Test that nested context managers work correctly."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("outer") as ctx_outer:
                y = x * 2
                with record_function_with_backward("inner") as ctx_inner:
                    z = y + 1
                    z = ctx_inner.mark(z)
                z = ctx_outer.mark(z)
            z.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "outer" in event_names
        assert "inner" in event_names
        assert "outer_backward" in event_names
        assert "inner_backward" in event_names


class TestMarkBackward:
    """Tests for the mark_backward function."""

    def test_backward_label_appears(self):
        """Test that the backward label appears in profiler trace."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            y = x * 2
            y = mark_backward(y, "my_op")
            y.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "my_op_backward" in event_names

    def test_gradient_correctness(self):
        """Test that gradients are correct after marking."""
        x = torch.randn(10, 10, requires_grad=True)

        y = x**2
        y = mark_backward(y, "square_op")
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, 2 * x)

    def test_no_grad_returns_same_tensor(self):
        """Test that tensors without requires_grad are returned unchanged."""
        x = torch.randn(10, 10, requires_grad=False)
        result = mark_backward(x, "test")
        assert result is x

    def test_tensor_values_unchanged(self):
        """Test that the tensor values are not modified."""
        x = torch.randn(5, 5, requires_grad=True)
        y = x * 2
        y_marked = mark_backward(y, "test")

        assert torch.allclose(y, y_marked)


class TestRecordFunctionWithBackwardFullBlock:
    """Tests for the mark_input/mark_output pattern that covers entire backward blocks."""

    def test_backward_label_covers_full_block(self):
        """Test that using mark_input and mark_output creates a backward region."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("full_block") as ctx:
                x_in = ctx.mark_input(x)
                # Multiple operations in the block
                y = x_in * 2
                y = y + 1
                y = y.relu()
                y = ctx.mark_output(y)
            y.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "full_block" in event_names
        assert "full_block_backward" in event_names

    def test_gradient_flows_correctly_with_markers(self):
        """Test that gradients flow correctly through input/output markers."""
        x = torch.randn(10, 10, requires_grad=True)

        with record_function_with_backward("block") as ctx:
            x_in = ctx.mark_input(x)
            y = x_in * 3
            y = ctx.mark_output(y)

        y.sum().backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full_like(x.grad, 3.0))

    def test_multiple_inputs(self):
        """Test that multiple inputs can be marked."""
        x = torch.randn(10, 10, requires_grad=True)
        w = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("matmul_block") as ctx:
                x_in = ctx.mark_input(x)
                w_in = ctx.mark_input(w)
                y = x_in @ w_in
                y = ctx.mark_output(y)
            y.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "matmul_block_backward" in event_names

        # Check gradients still flow correctly
        assert x.grad is not None
        assert w.grad is not None

    def test_no_grad_tensors_pass_through(self):
        """Test that tensors without requires_grad pass through mark_input/output."""
        x = torch.randn(10, 10, requires_grad=False)

        with record_function_with_backward("block") as ctx:
            x_in = ctx.mark_input(x)
            y = x_in * 2
            y_out = ctx.mark_output(y)

        # No error and values unchanged
        assert torch.allclose(x, x_in)
        assert torch.allclose(y, y_out)

    def test_multiple_outputs(self):
        """Test that multiple outputs can be marked."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("multi_output_block") as ctx:
                x_in = ctx.mark_input(x)
                y1 = x_in * 2
                y2 = x_in + 1
                y1 = ctx.mark_output(y1)
                y2 = ctx.mark_output(y2)
            (y1.sum() + y2.sum()).backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "multi_output_block" in event_names
        assert "multi_output_block_backward" in event_names

        # Check gradient flows correctly
        assert x.grad is not None
        # grad should be 2 (from y1) + 1 (from y2) = 3
        assert torch.allclose(x.grad, torch.full_like(x.grad, 3.0))

    def test_multiple_inputs_and_outputs(self):
        """Test blocks with multiple inputs AND multiple outputs."""
        a = torch.randn(10, 10, requires_grad=True)
        b = torch.randn(10, 10, requires_grad=True)
        c = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("complex_block") as ctx:
                a_in = ctx.mark_input(a)
                b_in = ctx.mark_input(b)
                c_in = ctx.mark_input(c)
                # Complex computation with multiple outputs
                out1 = a_in * b_in
                out2 = b_in + c_in
                out3 = a_in - c_in
                out1 = ctx.mark_output(out1)
                out2 = ctx.mark_output(out2)
                out3 = ctx.mark_output(out3)
            (out1.sum() + out2.sum() + out3.sum()).backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "complex_block" in event_names
        assert "complex_block_backward" in event_names

        # All inputs should have gradients
        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None

    def test_nested_blocks_with_mark_input_output(self):
        """Test nested blocks using mark_input/mark_output pattern."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("outer") as ctx_outer:
                x_outer = ctx_outer.mark_input(x)
                y = x_outer * 2

                with record_function_with_backward("inner") as ctx_inner:
                    y_inner = ctx_inner.mark_input(y)
                    z = y_inner + 1
                    z = z.relu()
                    z = ctx_inner.mark_output(z)

                result = z * 3
                result = ctx_outer.mark_output(result)
            result.sum().backward()

        event_names = [e.key for e in prof.key_averages()]
        assert "outer" in event_names
        assert "inner" in event_names
        assert "outer_backward" in event_names
        assert "inner_backward" in event_names

        # Gradient should flow correctly
        assert x.grad is not None

    def test_sequential_blocks(self):
        """Test multiple sequential blocks (like in escn_md_block.py)."""
        x = torch.randn(10, 10, requires_grad=True)
        w1 = torch.randn(10, 10, requires_grad=True)
        w2 = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            # Block 1: matmul
            with record_function_with_backward("block1_matmul") as ctx1:
                x_in = ctx1.mark_input(x)
                w1_in = ctx1.mark_input(w1)
                y = x_in @ w1_in
                y = ctx1.mark_output(y)

            # Block 2: activation
            with record_function_with_backward("block2_activation") as ctx2:
                y_in = ctx2.mark_input(y)
                z = y_in.relu()
                z = ctx2.mark_output(z)

            # Block 3: another matmul
            with record_function_with_backward("block3_matmul") as ctx3:
                z_in = ctx3.mark_input(z)
                w2_in = ctx3.mark_input(w2)
                out = z_in @ w2_in
                out = ctx3.mark_output(out)

            out.sum().backward()

        event_names = [e.key for e in prof.key_averages()]

        # All forward labels should appear
        assert "block1_matmul" in event_names
        assert "block2_activation" in event_names
        assert "block3_matmul" in event_names

        # All backward labels should appear
        assert "block1_matmul_backward" in event_names
        assert "block2_activation_backward" in event_names
        assert "block3_matmul_backward" in event_names

        # All gradients should be computed
        assert x.grad is not None
        assert w1.grad is not None
        assert w2.grad is not None

    def test_deeply_nested_blocks(self):
        """Test deeply nested blocks (3 levels)."""
        x = torch.randn(10, 10, requires_grad=True)

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function_with_backward("level1") as ctx1:
                x1 = ctx1.mark_input(x)
                y1 = x1 * 2

                with record_function_with_backward("level2") as ctx2:
                    y2 = ctx2.mark_input(y1)
                    z2 = y2 + 1

                    with record_function_with_backward("level3") as ctx3:
                        z3 = ctx3.mark_input(z2)
                        out3 = z3.relu()
                        out3 = ctx3.mark_output(out3)

                    out2 = out3 * 3
                    out2 = ctx2.mark_output(out2)

                out1 = out2 - 1
                out1 = ctx1.mark_output(out1)

            out1.sum().backward()

        event_names = [e.key for e in prof.key_averages()]

        # All levels should have forward and backward labels
        for level in ["level1", "level2", "level3"]:
            assert level in event_names, f"{level} not in events"
            assert f"{level}_backward" in event_names, f"{level}_backward not in events"

        assert x.grad is not None
