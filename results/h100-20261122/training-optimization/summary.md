# Training optimizations on H100

All standalone timings use rank 32. Three warm-ups precede 12 wall-clock measurements. A separate validation trajectory compares every expert loss and hashes every master and EMA parameter after each of the same 15 updates. The selection uses the lowest median among exact matches.

| Mode | Experts | Path | Train threads | Median ms | Expert steps/s | CPU % of one core | Exact trajectory |
|---|---:|---|---:|---:|---:|---:|---|
| cache | 1 | eager | 1 | 303.34 | 3.46 | 100.0 | yes |
| cache | 1 | eager | 4 | 271.58 | 3.56 | 129.4 | yes |
| cache | 1 | fullgraph | 1 | 58.86 | 16.98 | 100.1 | no |
| cache | 1 | graph | 1 | 59.48 | 16.81 | 100.1 | yes |
| cache | 1 | graph | 4 | 60.80 | 16.40 | 178.1 | yes |
| cache | 1 | packed | 1 | 539.10 | 1.95 | 100.1 | yes |
| cache | 1 | packed | 4 | 440.02 | 2.27 | 112.5 | yes |
| cache | 2 | eager | 1 | 298.35 | 6.21 | 100.0 | yes |
| cache | 2 | graph | 1 | 73.84 | 26.91 | 100.1 | yes |
| cache | 2 | graph | 4 | 73.27 | 27.22 | 165.8 | yes |
| cache | 4 | eager | 1 | 547.00 | 7.31 | 100.1 | yes |
| cache | 4 | graph | 1 | 96.14 | 41.49 | 100.1 | yes |
| cache | 4 | graph | 4 | 94.21 | 39.94 | 147.1 | yes |
| cache | 8 | eager | 1 | 620.22 | 12.83 | 100.0 | yes |
| cache | 8 | eager | 4 | 795.15 | 10.63 | 177.8 | yes |
| cache | 8 | fullgraph | 1 | 126.93 | 62.99 | 100.1 | no |
| cache | 8 | graph | 1 | 141.69 | 56.45 | 100.1 | yes |
| cache | 8 | graph | 4 | 139.82 | 56.63 | 135.1 | yes |
| cache | 8 | packed | 1 | 653.84 | 12.12 | 100.1 | yes |
| cache | 8 | packed | 4 | 382.57 | 20.84 | 112.6 | yes |
| recompute | 1 | eager | 1 | 359.01 | 2.78 | 100.0 | yes |
| recompute | 1 | eager | 4 | 410.34 | 2.39 | 122.8 | yes |
| recompute | 1 | fullgraph | 1 | 176.29 | 5.63 | 100.1 | no |
| recompute | 1 | graph | 1 | 164.96 | 6.03 | 100.1 | yes |
| recompute | 1 | graph | 4 | 166.60 | 6.00 | 132.8 | yes |
| recompute | 1 | packed | 1 | 364.79 | 2.68 | 100.0 | yes |
| recompute | 1 | packed | 4 | 371.92 | 2.62 | 114.3 | yes |
| recompute | 8 | eager | 1 | 1566.29 | 4.93 | 100.0 | yes |
| recompute | 8 | eager | 4 | 1785.13 | 4.50 | 128.6 | yes |
| recompute | 8 | fullgraph | 1 | 995.63 | 8.04 | 100.1 | no |
| recompute | 8 | graph | 1 | 1006.45 | 7.95 | 100.1 | yes |
| recompute | 8 | graph | 4 | 1010.73 | 7.89 | 105.5 | yes |
| recompute | 8 | packed | 1 | 1439.14 | 5.55 | 100.1 | yes |
| recompute | 8 | packed | 4 | 1337.05 | 5.94 | 104.2 | yes |

Packed input assembly transfers inputs and builds batches on the GPU. Prefix records are packed once into pinned buffers by dtype. Identical input objects shared by multiple experts are assembled once per update. Recompute still runs the VLM prefix for each expert and each microbatch on every step.

Graph timing includes input/RNG staging and replay. The graph path captures both forward/backward microbatches and on-device recomputation; the existing optimizer and finite-gradient checks run eagerly. Fullgraph also captures gradient clipping, capturable AdamW, and EMA, with tensor-valued learning rates. Step-dependent RNG and learning rate values are supplied before replay. Exactpack/exactgraph retain the original CPU numerical preprocessing and assemble batches on the GPU.

## CPU stacks

Top Python-defined functions by self wall time in one profiled production step. Self time includes native calls and waits reached from that function. Nested Python-defined calls are attributed to the nested frame. Timings in the main table are collected separately without profiling.

### cache, 1 expert(s)

| Python function | Calls | Self ms | Inclusive ms |
|---|---:|---:|---:|
| torch/autograd/graph.py(860): _engine_run_backward | 2 | 331.43 | 331.44 |
| evoserve/backend.py(38): forward | 218 | 26.91 | 41.60 |
| rlinf/models/embodiment/openpi/modules/gemma.py(679): _apply_rope_eager | 72 | 21.34 | 23.81 |
| rlinf/models/embodiment/openpi/modules/lora.py(195): _dot | 108 | 21.19 | 21.34 |
| rlinf/models/embodiment/openpi/modules/gemma.py(149): forward | 74 | 19.49 | 36.90 |
| rlinf/models/embodiment/openpi/modules/gemma.py(282): forward | 36 | 17.01 | 81.74 |
| torch/nn/modules/linear.py(130): forward | 226 | 14.99 | 15.09 |
| PIL/Image.py(2332): resize | 24 | 14.77 | 14.91 |
| torch/functional.py(176): einsum | 74 | 11.77 | 11.84 |
| evoserve/backend.py(253): <dictcomp> | 2 | 10.26 | 16.84 |

| Native PyTorch operation | Calls | Self ms |
|---|---:|---:|
| aten::bmm | 2264 | 48.34 |
| aten::mul | 4745 | 29.83 |
| aten::copy_ | 3293 | 29.49 |
| aten::mm | 1638 | 17.29 |
| aten::cat | 1927 | 17.22 |
| aten::add | 3177 | 13.54 |
| autograd::engine::evaluate_function: ViewBackward0 | 2580 | 13.37 |
| aten::view | 10675 | 12.50 |
| aten::empty_strided | 3103 | 12.04 |
| aten::add_ | 1616 | 10.61 |

### cache, 8 expert(s)

| Python function | Calls | Self ms | Inclusive ms |
|---|---:|---:|---:|
| torch/autograd/graph.py(860): _engine_run_backward | 2 | 346.70 | 346.71 |
| PIL/Image.py(2332): resize | 192 | 84.52 | 85.30 |
| evoserve/backend.py(253): <listcomp> | 596 | 47.52 | 47.52 |
| evoserve/backend.py(236): <lambda> | 3240 | 43.11 | 43.14 |
| evoserve/backend.py(231): step | 1 | 42.77 | 1006.90 |
| evoserve/backend.py(38): forward | 218 | 31.14 | 47.52 |
| rlinf/models/embodiment/openpi/modules/model.py(134): from_dict | 64 | 27.57 | 27.63 |
| evoserve/backend.py(91): <lambda> | 576 | 27.20 | 27.20 |
| rlinf/models/embodiment/openpi/modules/lora.py(195): _dot | 108 | 23.16 | 23.30 |
| rlinf/models/embodiment/openpi/modules/gemma.py(679): _apply_rope_eager | 72 | 20.88 | 23.38 |

| Native PyTorch operation | Calls | Self ms |
|---|---:|---:|
| aten::copy_ | 14969 | 101.44 |
| aten::bmm | 2264 | 44.60 |
| aten::empty_strided | 16291 | 42.89 |
| aten::add_ | 7874 | 29.77 |
| aten::mul | 4920 | 27.92 |
| aten::_to_copy | 13085 | 25.99 |
| aten::cat | 2263 | 19.69 |
| aten::add | 3184 | 16.63 |
| aten::mm | 1638 | 15.70 |
| aten::mul_ | 2384 | 12.34 |

### recompute, 1 expert(s)

| Python function | Calls | Self ms | Inclusive ms |
|---|---:|---:|---:|
| torch/autograd/graph.py(860): _engine_run_backward | 2 | 228.91 | 228.92 |
| rlinf/models/embodiment/openpi/modules/gemma.py(282): forward | 72 | 60.39 | 150.49 |
| rlinf/models/embodiment/openpi/modules/gemma.py(679): _apply_rope_eager | 144 | 32.88 | 37.37 |
| rlinf/models/embodiment/openpi/modules/lora.py(195): _dot | 108 | 28.30 | 28.43 |
| torch/nn/modules/linear.py(130): forward | 700 | 25.91 | 26.23 |
| evoserve/backend.py(38): forward | 218 | 25.89 | 39.84 |
| rlinf/models/embodiment/openpi/modules/gemma.py(149): forward | 148 | 24.92 | 42.65 |
| torch/nn/modules/activation.py(1255): forward | 162 | 23.69 | 25.76 |
| torch/functional.py(176): einsum | 146 | 18.96 | 19.07 |
| PIL/Image.py(2332): resize | 24 | 13.95 | 14.07 |

| Native PyTorch operation | Calls | Self ms |
|---|---:|---:|
| aten::copy_ | 3689 | 61.59 |
| aten::bmm | 2660 | 44.33 |
| aten::mul | 5329 | 36.44 |
| aten::mm | 2052 | 21.09 |
| aten::cat | 2074 | 17.64 |
| aten::add | 3799 | 17.25 |
| aten::view | 14671 | 11.60 |
| aten::empty_strided | 3259 | 10.71 |
| autograd::engine::evaluate_function: ViewBackward0 | 2580 | 8.84 |
| aten::add_ | 1622 | 8.13 |

### recompute, 8 expert(s)

| Python function | Calls | Self ms | Inclusive ms |
|---|---:|---:|---:|
| torch/autograd/graph.py(860): _engine_run_backward | 2 | 531.78 | 531.79 |
| rlinf/models/embodiment/openpi/modules/gemma.py(282): forward | 324 | 374.28 | 655.88 |
| torch/nn/modules/activation.py(1255): forward | 1296 | 199.44 | 217.45 |
| rlinf/models/embodiment/openpi/modules/gemma.py(679): _apply_rope_eager | 648 | 120.79 | 139.09 |
| torch/nn/modules/linear.py(130): forward | 4018 | 109.54 | 111.18 |
| PIL/Image.py(2332): resize | 192 | 84.10 | 84.84 |
| rlinf/models/embodiment/openpi/modules/gemma.py(149): forward | 666 | 75.34 | 104.02 |
| torch/functional.py(176): einsum | 650 | 67.63 | 68.10 |
| evoserve/backend.py(231): step | 1 | 57.67 | 2273.13 |
| torch/nn/functional.py(2914): layer_norm | 2640 | 54.91 | 55.51 |

| Native PyTorch operation | Calls | Self ms |
|---|---:|---:|
| aten::copy_ | 18137 | 444.50 |
| aten::bmm | 5432 | 107.56 |
| aten::mul | 9592 | 68.53 |
| aten::mm | 4950 | 67.52 |
| aten::addmm | 4100 | 60.17 |
| aten::add | 8160 | 55.76 |
| aten::empty_strided | 17539 | 52.47 |
| aten::add_ | 7922 | 38.69 |
| aten::empty | 13586 | 34.66 |
| aten::cat | 3439 | 33.48 |
