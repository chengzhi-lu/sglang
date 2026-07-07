import sys

import pytest
import torch
from sgl_kernel.kvcacheio import (
    layerkv_copy_kv_span_backup_batched,
    layerkv_copy_kv_span_scatter,
    layerkv_copy_kv_span_scatter_batched,
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
)

from sglang.srt.utils import get_cuda_version, is_hip

# Skip entire module on CUDA 13.x — segfaults in transfer_kv kernel.
# Reference failure: https://github.com/sgl-project/sglang/actions/runs/24600433057/job/71938317621?pr=23119
pytestmark = pytest.mark.skipif(
    get_cuda_version()[0] >= 13,
    reason="test_kvcacheio segfaults on CUDA 13.x (sgl-kernel bug)",
)


def ref_copy_with_indices(src_pool, dst_pool, src_indices, dst_indices):
    dst_pool[dst_indices] = src_pool[src_indices].to(dst_pool.device)


def ref_copy_with_indices_pf_direct(
    src_pool, dst_pool, src_indices, dst_indices, page_size, layer_id, lf_to_pf=False
):
    if lf_to_pf:
        for i in range(0, len(src_indices), page_size):
            dst_pool[dst_indices[i] // page_size][layer_id] = src_pool[layer_id][
                src_indices[i : i + page_size]
            ].to(dst_pool.device)
    else:
        for i in range(0, len(src_indices), page_size):
            dst_pool[layer_id][dst_indices[i : i + page_size]] = src_pool[
                src_indices[i] // page_size
            ][layer_id].to(dst_pool.device)


def ref_copy_with_indices_page_head(
    src_pool,
    dst_pool,
    src_indices,
    dst_indices,
    page_size,
    layer_id,
    head_num,
    lf_to_ph=False,
):
    if lf_to_ph:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[dst_indices[i] // page_size][head_id][
                    dst_indices[i] % page_size
                ][layer_id] = src_pool[layer_id][src_indices[i]][head_id].to(
                    dst_pool.device
                )
    else:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[layer_id][dst_indices[i]][head_id] = src_pool[
                    src_indices[i] // page_size
                ][head_id][src_indices[i] % page_size][layer_id].to(dst_pool.device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [1, 128, 1024])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("all_layers", [False, True])
def test_transfer_kv(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    all_layers: bool,
):
    """
    Tests the per-layer transfer functions, treating tensors as memory pools.
    """

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4  # A small number of layers for pool creation

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # Prepare memory pools based on whether it's an MLA case.
    if is_mla:
        src_pool_host = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_pool_ref = torch.zeros_like(src_pool_host).to(device)
        dst_pool_kernel = torch.zeros_like(dst_pool_ref)
        dst_pool_direct = torch.zeros_like(dst_pool_ref)
    else:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_k_pool_ref = torch.zeros_like(src_k_pool).to(device)
        dst_v_pool_ref = torch.zeros_like(src_v_pool).to(device)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)

    torch.cuda.synchronize()

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if is_mla:
        if not all_layers:
            ref_copy_with_indices(
                src_pool_host[layer_idx_to_test],
                dst_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer_mla(
                src_pool_host[layer_idx_to_test],
                dst_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_pool_host[layer_idx_to_test]],
                [dst_pool_direct[layer_idx_to_test]],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_pool_host[layer_id],
                    dst_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
            src_layers_device = torch.tensor(
                [src_pool_host[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_layers_device = torch.tensor(
                [
                    dst_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla(
                src_layers_device,
                dst_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_pool_host[layer_id] for layer_id in range(num_layers)],
                [dst_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_pool_kernel, dst_pool_ref)
        torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
    else:
        if not all_layers:
            ref_copy_with_indices(
                src_k_pool[layer_idx_to_test],
                dst_k_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            ref_copy_with_indices(
                src_v_pool[layer_idx_to_test],
                dst_v_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer(
                src_k_pool[layer_idx_to_test],
                dst_k_pool_kernel[layer_idx_to_test],
                src_v_pool[layer_idx_to_test],
                dst_v_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_k_pool[layer_idx_to_test], src_v_pool[layer_idx_to_test]],
                [
                    dst_k_pool_direct[layer_idx_to_test],
                    dst_v_pool_direct[layer_idx_to_test],
                ],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_k_pool[layer_id],
                    dst_k_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
                ref_copy_with_indices(
                    src_v_pool[layer_id],
                    dst_v_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )

            src_k_layers_device = torch.tensor(
                [src_k_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            src_v_layers_device = torch.tensor(
                [src_v_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_k_layers_device = torch.tensor(
                [
                    dst_k_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            dst_v_layers_device = torch.tensor(
                [
                    dst_v_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer(
                src_k_layers_device,
                dst_k_layers_device,
                src_v_layers_device,
                dst_v_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_k_pool[layer_id] for layer_id in range(num_layers)]
                + [src_v_pool[layer_id] for layer_id in range(num_layers)],
                [dst_k_pool_direct[layer_id] for layer_id in range(num_layers)]
                + [dst_v_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
        torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)

    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [128, 1024, 8192])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [20480])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("lf_to_pf", [False, True])
def test_transfer_kv_pf_direct(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    lf_to_pf: bool,
):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)
    test_stream = torch.cuda.Stream()

    num_layers = 4

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if lf_to_pf:
        if is_mla:
            src_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_pool_ptrs = [src_pool[i] for i in range(num_layers)]
            dst_pool_ref = torch.zeros(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            dst_pool_direct = torch.zeros_like(dst_pool_ref)
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_all_layer_direct_lf_pf(
                    src_pool_ptrs,
                    [dst_pool_direct],
                    src_indices_host,
                    dst_indices_host,
                    page_size,
                )
            test_stream.synchronize()

            for i in range(num_layers):
                ref_copy_with_indices_pf_direct(
                    src_pool,
                    dst_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_pool_direct, dst_pool_ref)

        else:
            src_k_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_k_pool_ptrs = [src_k_pool[i] for i in range(num_layers)]
            src_v_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_v_pool_ptrs = [src_v_pool[i] for i in range(num_layers)]
            dst_k_pool_ref = torch.zeros(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
            dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
            dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_all_layer_direct_lf_pf(
                    src_k_pool_ptrs + src_v_pool_ptrs,
                    [dst_k_pool_direct, dst_v_pool_direct],
                    src_indices_host,
                    dst_indices_host,
                    page_size,
                )
            test_stream.synchronize()

            for i in range(num_layers):
                ref_copy_with_indices_pf_direct(
                    src_k_pool,
                    dst_k_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
                ref_copy_with_indices_pf_direct(
                    src_v_pool,
                    dst_v_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
            torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)
    else:
        if is_mla:
            src_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()

            dst_pool_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(
                device
            )
            dst_pool_direct = torch.zeros_like(dst_pool_ref)
            dst_pool_direct_ptrs = [dst_pool_direct[i] for i in range(num_layers)]
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_per_layer_direct_pf_lf(
                    [src_pool],
                    [dst_pool_direct_ptrs[layer_idx_to_test]],
                    src_indices_host,
                    dst_indices_host,
                    layer_idx_to_test,
                    page_size,
                )
            test_stream.synchronize()

            ref_copy_with_indices_pf_direct(
                src_pool,
                dst_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
        else:
            src_k_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            src_v_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()

            dst_k_pool_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(
                device
            )
            dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
            dst_k_pool_direct_ptrs = [dst_k_pool_direct[i] for i in range(num_layers)]

            dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
            dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)
            dst_v_pool_direct_ptrs = [dst_v_pool_direct[i] for i in range(num_layers)]
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_per_layer_direct_pf_lf(
                    [src_k_pool, src_v_pool],
                    [
                        dst_k_pool_direct_ptrs[layer_idx_to_test],
                        dst_v_pool_direct_ptrs[layer_idx_to_test],
                    ],
                    src_indices_host,
                    dst_indices_host,
                    layer_idx_to_test,
                    page_size,
                )
            test_stream.synchronize()

            ref_copy_with_indices_pf_direct(
                src_k_pool,
                dst_k_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )
            ref_copy_with_indices_pf_direct(
                src_v_pool,
                dst_v_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )

            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
            torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)
    torch.set_default_dtype(original_dtype)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [1024])
@pytest.mark.parametrize("head_num", [8, 16])
@pytest.mark.parametrize("total_items_in_pool", [4096])
@pytest.mark.parametrize("lf_to_ph", [False, True])
def test_transfer_kv_page_head(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    page_size: int,
    item_size: int,
    head_num: int,
    total_items_in_pool: int,
    lf_to_ph: bool,
):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    assert item_size % head_num == 0
    head_dim = item_size // head_num

    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if lf_to_ph:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        src_k_pool_ptrs = [src_k_pool[i] for i in range(num_layers)]
        src_k_pool_ptrs = torch.tensor(
            [x.data_ptr() for x in src_k_pool_ptrs],
            dtype=torch.uint64,
            device=device,
        )
        src_v_pool_ptrs = [src_v_pool[i] for i in range(num_layers)]
        src_v_pool_ptrs = torch.tensor(
            [x.data_ptr() for x in src_v_pool_ptrs],
            dtype=torch.uint64,
            device=device,
        )

        dst_k_pool_ref = torch.zeros(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()
        dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref).pin_memory()

        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref).pin_memory()
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref).pin_memory()
        torch.cuda.synchronize()

        transfer_kv_all_layer_lf_ph(
            src_k_pool_ptrs,
            dst_k_pool_kernel,
            src_v_pool_ptrs,
            dst_v_pool_kernel,
            src_indices_device,
            dst_indices_device,
            item_size * dtype.itemsize,
            item_size * num_layers * dtype.itemsize,
            num_layers,
            page_size,
            head_num,
        )
        torch.cuda.synchronize()

        for i in range(num_layers):
            ref_copy_with_indices_page_head(
                src_k_pool,
                dst_k_pool_ref,
                src_indices_device,
                dst_indices_host,
                page_size,
                i,
                head_num,
                lf_to_ph=True,
            )
            ref_copy_with_indices_page_head(
                src_v_pool,
                dst_v_pool_ref,
                src_indices_device,
                dst_indices_host,
                page_size,
                i,
                head_num,
                lf_to_ph=True,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
    else:
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_ph_lf

        src_k_pool = torch.randn(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()
        src_v_pool = torch.randn(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()

        dst_k_pool_ref = torch.zeros(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_kernel_ptrs = [dst_k_pool_kernel[i] for i in range(num_layers)]
        dst_v_pool_kernel_ptrs = [dst_v_pool_kernel[i] for i in range(num_layers)]
        torch.cuda.synchronize()

        transfer_kv_per_layer_ph_lf(
            src_k_pool,
            dst_k_pool_kernel_ptrs[layer_idx_to_test],
            src_v_pool,
            dst_v_pool_kernel_ptrs[layer_idx_to_test],
            src_indices_device,
            dst_indices_device,
            layer_idx_to_test,
            item_size * dtype.itemsize,
            item_size * num_layers * dtype.itemsize,
            page_size,
            head_num,
        )

        ref_copy_with_indices_page_head(
            src_k_pool,
            dst_k_pool_ref,
            src_indices_host,
            dst_indices_device,
            page_size,
            layer_idx_to_test,
            head_num,
            lf_to_ph=False,
        )
        ref_copy_with_indices_page_head(
            src_v_pool,
            dst_v_pool_ref,
            src_indices_host,
            dst_indices_device,
            page_size,
            layer_idx_to_test,
            head_num,
            lf_to_ph=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(32,), (4, 16)])
def test_layerkv_copy_kv_span_scatter(dtype: torch.dtype, shape: tuple[int, ...]):
    device = "cuda"
    src_base_slot = 10
    dst_base_slot = 64
    host_first = src_base_slot
    host_last = 42
    total_dst_tokens = 128
    spans = ((10, 0, 3), (18, 3, 5), (31, 8, 4))
    token_count = sum(length for _start, _offset, length in spans)

    src_shape = (host_last - host_first, *shape)
    dst_shape = (total_dst_tokens, *shape)
    src_k = torch.randn(src_shape, dtype=dtype, device=device)
    src_v = torch.randn(src_shape, dtype=dtype, device=device)
    dst_k = torch.zeros(dst_shape, dtype=dtype, device=device)
    dst_v = torch.zeros(dst_shape, dtype=dtype, device=device)
    ref_k = torch.zeros_like(dst_k)
    ref_v = torch.zeros_like(dst_v)

    spans_tensor = torch.tensor(spans, dtype=torch.int64, device=device)
    item_size = src_k[0].numel() * src_k.element_size()
    layerkv_copy_kv_span_scatter(
        src_k,
        src_v,
        dst_k,
        dst_v,
        spans_tensor,
        src_base_slot,
        dst_base_slot,
        item_size,
        8,
    )

    for host_start, scratch_offset, length in spans:
        src_slice = slice(
            host_start - src_base_slot, host_start - src_base_slot + length
        )
        dst_slice = slice(
            dst_base_slot + scratch_offset,
            dst_base_slot + scratch_offset + length,
        )
        ref_k[dst_slice].copy_(src_k[src_slice])
        ref_v[dst_slice].copy_(src_v[src_slice])

    torch.cuda.synchronize()
    torch.testing.assert_close(dst_k, ref_k)
    torch.testing.assert_close(dst_v, ref_v)
    assert token_count == 12


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(32,), (4, 16)])
def test_layerkv_copy_kv_span_scatter_batched(
    dtype: torch.dtype, shape: tuple[int, ...]
):
    device = "cuda"
    spans = ((10, 0, 3), (18, 3, 5), (31, 8, 4), (70, 0, 6), (81, 6, 2))
    span_batch_ids = (0, 0, 0, 1, 1)
    src_base_slots = (10, 70)
    dst_base_slots = (64, 96)

    src_ks = [
        torch.randn((40, *shape), dtype=dtype, device="cpu").pin_memory(),
        torch.randn((32, *shape), dtype=dtype, device="cpu").pin_memory(),
    ]
    src_vs = [
        torch.randn((40, *shape), dtype=dtype, device="cpu").pin_memory(),
        torch.randn((32, *shape), dtype=dtype, device="cpu").pin_memory(),
    ]
    dst_ks = [torch.zeros((128, *shape), dtype=dtype, device=device) for _ in range(2)]
    dst_vs = [torch.zeros((128, *shape), dtype=dtype, device=device) for _ in range(2)]
    ref_ks = [torch.zeros_like(dst) for dst in dst_ks]
    ref_vs = [torch.zeros_like(dst) for dst in dst_vs]

    layerkv_copy_kv_span_scatter_batched(
        src_ks,
        src_vs,
        dst_ks,
        dst_vs,
        torch.tensor(spans, dtype=torch.int64, device="cpu"),
        torch.tensor(span_batch_ids, dtype=torch.int64, device="cpu"),
        torch.tensor(src_base_slots, dtype=torch.int64, device="cpu"),
        torch.tensor(dst_base_slots, dtype=torch.int64, device="cpu"),
        src_ks[0][0].numel() * src_ks[0].element_size(),
        8,
    )

    for span, batch_id in zip(spans, span_batch_ids):
        host_start, scratch_offset, length = span
        src_slice = slice(
            host_start - src_base_slots[batch_id],
            host_start - src_base_slots[batch_id] + length,
        )
        dst_slice = slice(
            dst_base_slots[batch_id] + scratch_offset,
            dst_base_slots[batch_id] + scratch_offset + length,
        )
        ref_ks[batch_id][dst_slice].copy_(src_ks[batch_id][src_slice])
        ref_vs[batch_id][dst_slice].copy_(src_vs[batch_id][src_slice])

    torch.cuda.synchronize()
    for got, ref in zip(dst_ks, ref_ks):
        torch.testing.assert_close(got, ref)
    for got, ref in zip(dst_vs, ref_vs):
        torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(32,), (4, 16)])
def test_layerkv_copy_kv_span_backup_batched(
    dtype: torch.dtype, shape: tuple[int, ...]
):
    device = "cuda"
    spans = ((2, 11, 4), (9, 21, 3), (20, 5, 6), (4, 31, 5))
    span_batch_ids = (0, 0, 1, 1)
    src_ks = [torch.randn((64, *shape), dtype=dtype, device=device) for _ in range(2)]
    src_vs = [torch.randn((64, *shape), dtype=dtype, device=device) for _ in range(2)]
    dst_ks = [
        torch.zeros((64, *shape), dtype=dtype, device="cpu").pin_memory()
        for _ in range(2)
    ]
    dst_vs = [
        torch.zeros((64, *shape), dtype=dtype, device="cpu").pin_memory()
        for _ in range(2)
    ]

    layerkv_copy_kv_span_backup_batched(
        src_ks,
        src_vs,
        dst_ks,
        dst_vs,
        torch.tensor(spans, dtype=torch.int64, device="cpu"),
        torch.tensor(span_batch_ids, dtype=torch.int64, device="cpu"),
        src_ks[0][0].numel() * src_ks[0].element_size(),
    )

    torch.cuda.synchronize()
    for span, batch_id in zip(spans, span_batch_ids):
        src_start, dst_start, length = span
        src_slice = slice(src_start, src_start + length)
        dst_slice = slice(dst_start, dst_start + length)
        torch.testing.assert_close(
            dst_ks[batch_id][dst_slice], src_ks[batch_id][src_slice].cpu()
        )
        torch.testing.assert_close(
            dst_vs[batch_id][dst_slice], src_vs[batch_id][src_slice].cpu()
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
