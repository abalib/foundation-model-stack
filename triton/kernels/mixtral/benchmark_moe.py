import pytest
import torch
import triton
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul
from v0_moe_fused import fused_moe as fused_moe_base
from splitk_moe_fused import fused_moe as fused_moe_splitk_col_major
import time

def torch_moe(a, w1, w2, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[1],
                      dtype=a.dtype,
                      device=a.device)
    
    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1)).sum(dim=1)


def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.cuda.manual_seed(3227)
    
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10
    
    score = torch.randn((m, e), device='cuda', dtype=dtype)
    score = torch.softmax(score, dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    
    
    triton_output_splitk_col_major = fused_moe_splitk_col_major(a, w1, w2, topk_weight, topk_ids, False)
    # triton_output_splitk_linear = fused_moe_splitk_linear(a, w1, w2, topk_weight, topk_ids, False)
    triton_output_base = fused_moe_base(a, w1, w2, topk_weight, topk_ids, False)

    # breakpoint()

    # torch_base = torch_moe(a, w1, w2, topk_weight, topk_ids)
    # torch.testing.assert_close(triton_output_base, torch_base, atol=0.0, rtol=1e-1)


if __name__ == '__main__':

    # test_fused_moe(2, 14336//2, 4096, 8, 2, torch.float16)


    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['m'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            2**i for i in range(0, 10)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['spk', 'dp'],
        # Label name for the lines
        line_names=["SplitK Fused MoE GEMM Kernel", "Data Parallel Fused MoE GEMM Kernel"],

        # Line styles
        styles=[('blue', '-'), ('green', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="test",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
    def benchmark(m, provider):
        
        m = m
        n = 14336//2
        k = 4096
        e = 8
        topk = 2

        torch.cuda.manual_seed(3227)
        dtype = torch.float16

        a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10
        
        score = torch.randn((m, e), device='cuda', dtype=dtype)
        score = torch.softmax(score, dim=-1)
        topk_weight, topk_ids = torch.topk(score, topk)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'spk':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_moe_splitk_col_major(a, w1, w2, topk_weight, topk_ids, False), quantiles=quantiles)
        if provider == 'dp':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_moe_base(a, w1, w2, topk_weight, topk_ids, False), quantiles=quantiles)
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path='/nvme2/adhoq26/foundation-model-stack/triton/quantization/mixtral')