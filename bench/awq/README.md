# Benchmark scrip to evaluate AWQ qint4 kernels

This is a copy from the benchmark script in [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

## Prerequisites

```shell
$ pip install autoawq tabulate
```

## Quantize a model

```shell
$ python quantize.py --model_id princeton-nlp/Sheared-LLaMA-1.3B \
                     --version GEMV                              \
                     --group_size 64
```

## Evaluate performance vs float

FP16:

```shell
$ python quantize.py --model_path princeton-nlp/Sheared-LLaMA-1.3B
```

Quantized:

```shell
$ python benchmark.py --model_path ./princeton-nlp-Sheared-LLaMA-1.3B_64_GEMM
```
