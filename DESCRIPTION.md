**EARLY ACCESS:** The ZenDNN PyTorch* Plugin (zentorch) extends PyTorch* with an innovative upgrade that's set to revolutionize performance on AMD hardware.

As of version 4.2, AMD is unveiling a game-changing upgrade to ZenDNN, introducing a cutting-edge plug-in mechanism and an enhanced architecture under the hood. This isn't just about extensions; ZenDNN's aggressive AMD-specific optimizations operate at every level. It delves into comprehensive graph optimizations, including pattern identification, graph reordering, and seeking opportunities for graph fusions. At the operator level, ZenDNN boasts enhancements with microkernels, mempool optimizations, and efficient multi-threading on the large number of AMD EPYC cores. Microkernel optimizations further exploit all possible low-level math libraries, including AOCL BLIS.

The result? Enhanced performance with respect to baseline PyTorch*. zentorch leverages torch.compile, the latest PyTorch enhancement for accelerated performance. torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes and unlocking unprecedented speed and efficiency.

The ZenDNN PyTorch plugin is compatible with PyTorch version 2.1.2.

## Support

Please note that zentorch is currently in “Early Access” mode. We welcome feedback, suggestions, and bug reports. Should you have any of the these, please contact us on zendnn.maintainers@amd.com

## License

zentorch is licensed under [Apache-2.0, MIT, BSD-3-Clause](https://github.com/amd/ZenDNN-pytorch-plugin/blob/main/LICENSE).