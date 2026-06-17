__The latest ZenDNN Plugin for PyTorch* (zentorch) is here!__

The ZenDNN PyTorch plugin is called zentorch. Combined with PyTorch's torch.compile, zentorch transforms deep learning pipelines into finely-tuned, AMD-specific engines, delivering unparalleled efficiency and speed for large-scale inference workloads.

This upgrade continues the focus on optimizing inference with Recommender Systems and Large Language Models on AMD EPYC™ CPUs.
It includes AMD EPYC™ enhancements for bfloat16 performance, expanded support for cutting-edge models like Llama 3.2 and 3.3, Microsoft Phi, and more as well as support for a wide-variety of quantization configurations.
The quantization support included 4-bit weight-only quantization, along with support for INT8 dynamic activation and INT8 weight quantization, and quantized support for the DLRM-v2 model with a mix of 8-bit and 4-bit quantization.
This also includes support for running generative models with vLLM. This release introduces functional support for running LLMs using float16 precision with vLLM on 6th Gen AMD EPYC™ processors.

Under the hood, ZenDNN’s enhanced AMD-specific optimizations operate at every level. In addition to highly optimized operator microkernels, these include comprehensive graph optimizations including pattern identification, graph reordering, and fusions.
They also incorporate optimized embedding bag kernels and enhanced zenMatMul matrix splitting strategies which leverage the AMD EPYC™ microarchitecture to deliver enhanced throughput and latency.

The vLLM-ZenTorch plugin extends these benefits to the vLLM inference engine, enabling plug-and-play acceleration of large language model inference on AMD EPYC™ CPUs. By integrating ZenTorch with vLLM, users can achieve significant throughput improvements for LLM workloads with zero code changes.

The zentorch plugin seamlessly works with PyTorch {{PYTORCH_VERSION}}, offering a high-performance experience for deep learning on AMD EPYC™ platforms.

In addition to stable (GA) releases, the zentorch plugin provides weekly releases via the `zentorch-weekly` package on PyPI.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of the these, please kindly file an issue on the ZenDNN Plugin for PyTorch GitHub page [here](https://github.com/amd/ZenDNN-pytorch-plugin/issues)

## License

AMD copyrighted code in ZenDNN is subject to the [Apache-2.0, MIT, or BSD-3-Clause](https://github.com/amd/ZenDNN-pytorch-plugin/blob/main/LICENSE) licenses; consult the source code file headers for the applicable license.  Third party copyrighted code in ZenDNN is subject to the licenses set forth in the source code file headers of such code.
