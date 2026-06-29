"""llm-inference-performance.

A compact, dependency-light lab for LLM inference *performance* engineering:

    model           nanoGPT-style decoder with optional KV-cache
    generate        greedy decoding with cache on/off (the core A/B)
    batching        batched throughput scaling
    quantize        int8 dynamic quantization (size + latency)
    metrics         latency percentiles + device-aware memory probes
    bench           the benchmark harness that ties it together
    distributed     tensor-parallel linear layers (gloo on CPU, NCCL on GPU)

Weights are randomly initialized at a realistic shape: this measures the
*systems* behavior of serving (latency / throughput / memory), and correctness
tests assert the optimizations are numerically equivalent to the baseline.
"""

__version__ = "0.1.0"
