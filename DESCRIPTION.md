__The latest ZenDNN Plugin for PyTorch* (zentorch) 5.0.1 is here!__

ZenDNN 5.0.1 is a minor release building upon the major ZenDNN 5.0 release. This upgrade continues the focus on optimizing inference with Recommender Systems and Large Language Models on AMD EPYC™ CPUs.
ZenDNN includes AMD EPYC™ enhancements for bfloat16 performance, expanded support for cutting-edge models like Llama 3.1 and 3.2, Microsoft Phi, and more as well as support for INT4 quantized datatype.
This includes the advanced Activation-Aware Weight Quantization (AWQ) algorithm for LLMs and quantized support for the DLRM-v2 model with int8 weights.

Under the hood, ZenDNN’s enhanced AMD-specific optimizations operate at every level. In addition to highly optimized operator microkernels, these include comprehensive graph optimizations including pattern identification, graph reordering, and fusions.
They also incorporate optimized embedding bag kernels and enhanced zenMatMul matrix splitting strategies which leverage the AMD EPYC™ microarchitecture to deliver enhanced throughput and latency.

The ZenDNN PyTorch plugin is called zentorch. Combined with PyTorch's torch.compile, zentorch transforms deep learning pipelines into finely-tuned, AMD-specific engines, delivering unparalleled efficiency and speed for large-scale inference workloads.

The zentorch 5.0.1 release plugs seamlessly with PyTorch versions from 2.5 to 2.2, offering a high-performance experience for deep learning on AMD EPYC™ platforms.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of the these, please kindly file an issue on the ZenDNN Plugin for PyTorch Github page [here](https://github.com/amd/ZenDNN-pytorch-plugin/issues)

## License

AMD copyrighted code in ZenDNN is subject to the [Apache-2.0, MIT, or BSD-3-Clause](https://github.com/amd/ZenDNN-pytorch-plugin/blob/main/LICENSE) licenses; consult the source code file headers for the applicable license.  Third party copyrighted code in ZenDNN is subject to the licenses set forth in the source code file headers of such code.