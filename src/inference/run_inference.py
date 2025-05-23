from vllm import LLM, SamplingParams

prompts = ['{"content": "I am trying to write a function in Python that can show the details of a matrix in a readable format. Do you have any suggestions on how to write that function?","role": "user"}'
]
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, presence_penalty=0, frequency_penalty=0)

def main():
    llm = LLM(model="anatal/qwen2_05_smol-smoltalk", dtype="half")
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        for _output in output.outputs:
            generated_text = _output.text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)


if __name__ == "__main__":
    main()
