# ******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc.
# * All rights reserved.
# ******************************************************************************

import os

# Force the W4A16 path. Zentorch reads this while importing vllm and skips
# the DA8W4 patch when it is 0, so it must be set before that import.
os.environ["VLLM_CPU_INT4_W4A8"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    print("\n" + "=" * 10 + " LLM-Compressor w4a16 Example Execution Started " + "=" * 10 + "\n")
    model_id = "amd/Llama-3.1-8B-Instruct-w4a16-llmcompressor"
    print(f"Loading W4A16 quantized model: {model_id}")
    llm = LLM(model=model_id)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    print("Running inference")
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

    print("\n" + "=" * 10 + " Script Executed Successfully " + "=" * 10 + "\n")


if __name__ == "__main__":
    main()
