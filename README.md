# GraalVM Python performance

This project contains some performance benchmarks that can be run on GraalVM.

## Introduction

GraalVM supports Python, albeit in a limited fashion. This is best exemplified by the
message you get every time you start the GraalVM Python engine:

```
Please note: This Python implementation is in the very early stages, and can run
little more than basic benchmarks at this point.
```

They're not kidding. I've tried getting [pyperformance](https://github.com/python/performance/blob/master/pyperformance)
to run on GraalVM, but actually getting there required quite a bit of hacking.
The result of this is this project.

## Disclaimer

A lot of changes were required to be able to run pyperformance. As a result, the
information taken from this benchmark should not be taken seriously in any manner.
E.g. pyperformance makes use of low level, high accuracy, timing mechanisms which
are currently not available in GraalVM. As a result, this benchmark falls
back to `time.time()`, which is not ideal.

Also, many benchmarks wouldn't even run. I've got three working at the moment,
which are in this repository. Many other benchmarks either caused an internal
error GraalVM, or required missing functionality.

## Running the benchmark

The benchmark requires Linux to run. To run the benchmark, clone this project.
Then, one level up, extract PyPy3 from [https://pypy.org/download.html#default-with-a-jit-compiler]
and GraalVM from [https://www.graalvm.org/downloads/]. I've used EE, but CE should
work as well.

The `runall.sh` script will find the benchmarks and runtimes and run all benchmarks
present against all the runtimes.

## Results

These results were retrieved using:

* Python3: 3.6.5
* PyPy: v6.0.0
* GraalVM: EE 1.0.0-rc5

| Benchmark            | Python           | PyPy              | GraalVM           |
| -------------------- | ---------------- | ----------------- | ----------------- |
| `bm_fannkuch`        | 732 ms ± 3.95 ms | 119 ms ± 1.26 ms  | 416 ms ± 68.2 ms  |
| `bm_richards`        | 131 ms ± 1.09 ms | 11.4 ms ± 344 us  | 109 ms ± 31.5 ms  |
| `bm_spectral_norm`   | 191 ms ± 2.50 ms | 5.52 ms ± 68.8 us | 35.6 ms ± 12.9 ms |

And as a nice chart:

![Performance statistics](https://raw.githubusercontent.com/pvginkel/graalvm-python-performance/master/chart.png)
