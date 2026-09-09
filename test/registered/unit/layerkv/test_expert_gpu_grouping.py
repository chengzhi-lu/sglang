import pytest
import torch

from sglang.jit_kernel.layerkv_expert_group import (
    layerkv_group_token_experts,
    layerkv_group_token_experts_multi,
)
from sglang.srt.layerkv.residency_budget import group_token_experts
from sglang.srt.layerkv.shared_expert import SharedExpertController
from sglang.srt.layerkv.runtime import LayerKVConfig
from sglang.srt.layerkv.runtime import LayerKVRuntime


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gpu_grouping_matches_cpu_first_fit(dtype):
    routes = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [0, 1, 2], [1, 5, 6], [-1, 7, 99]],
        dtype=dtype,
        device="cuda",
    )
    actual, valid = layerkv_group_token_experts(routes, 4, 8)
    expected, expected_valid = group_token_experts(routes.cpu().tolist(), 4, 8)
    assert actual == expected
    assert valid == expected_valid


@cuda
def test_gpu_grouping_can_keep_row_indices_on_device():
    routes = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [0, 1, 2], [1, 5, 6]],
        dtype=torch.int64,
        device="cuda",
    )
    actual, valid = layerkv_group_token_experts(
        routes, 4, 8, device_rows=True
    )
    expected, expected_valid = group_token_experts(routes.cpu().tolist(), 4, 8)
    assert valid == expected_valid
    assert [mask for mask, _rows in actual] == [mask for mask, _rows in expected]
    assert all(rows.is_cuda for _mask, rows in actual)
    assert [rows.cpu().tolist() for _mask, rows in actual] == [
        rows for _mask, rows in expected
    ]


@cuda
def test_gpu_grouping_rejects_route_wider_than_capacity():
    routes = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="cannot cover"):
        layerkv_group_token_experts(routes, 2, 8)


@cuda
def test_gpu_grouping_multi_matches_each_cpu_first_fit():
    routes = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [0, 1, 2], [1, 5, 6]],
        dtype=torch.int64,
        device="cuda",
    )
    capacities = [4, 5, 7]
    actual = layerkv_group_token_experts_multi(
        routes, capacities, 8, device_rows=True
    )
    for (groups, valid), capacity in zip(actual, capacities):
        expected, expected_valid = group_token_experts(
            routes.cpu().tolist(), capacity, 8
        )
        assert valid == expected_valid
        assert [mask for mask, _rows in groups] == [
            mask for mask, _rows in expected
        ]
        assert all(rows.is_cuda for _mask, rows in groups)
        assert [rows.cpu().tolist() for _mask, rows in groups] == [
            rows for _mask, rows in expected
        ]


@cuda
def test_shared_controller_reuses_one_multi_capacity_pass():
    runtime = LayerKVRuntime(
        LayerKVConfig(
            enabled=True,
            mode="kvc-expert",
            policy="expert-first",
            shared_expert_layer=0,
            shared_expert_initial_slots=4,
            shared_expert_extra_slots=3,
            shared_expert_policy="adaptive",
            shared_expert_free_kv_donors=True,
            shared_expert_gpu_grouping=True,
        )
    )
    runtime._decode_step = 1
    state = type(
        "State",
        (),
        {
            "full_num_experts": 8,
            "slot_capacity": 4,
            "module": type("Module", (), {"top_k": 2})(),
        },
    )()
    controller = SharedExpertController(runtime, None)
    controller.state = state
    routes = torch.tensor(
        [[0, 1], [2, 3], [0, 1], [4, 5]],
        dtype=torch.int64,
        device="cuda",
    )

    actual, valid = controller.prepare_gpu_decode_groups(state, routes)
    expected, expected_valid = group_token_experts(routes.cpu().tolist(), 4, 8)
    assert valid == expected_valid
    assert [mask for mask, _rows in actual] == [
        mask for mask, _rows in expected
    ]
    assert controller.token_chunk_gpu_group_calls == 1
    assert controller.budget.last_observed_step == 1


def test_gpu_grouping_is_opt_in():
    assert not LayerKVConfig().shared_expert_gpu_grouping
