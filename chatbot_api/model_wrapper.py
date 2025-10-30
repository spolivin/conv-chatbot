import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class ChatModel:
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: torch.device = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Choosing params for configuring the model depending on the device
        if torch.cuda.is_available():
            load_kwargs = {"dtype": torch.float16, "device_map": "auto"}
        else:
            load_kwargs = {
                "dtype": torch.float32,
                "device_map": None,
                "low_cpu_mem_usage": True,
            }

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """Generates a response from the LLM based on the provided chat history.

        Args:
            messages (list[dict[str, str]]): Chat history.
            max_new_tokens (int, optional): Maximum number of tokens to generate in the output sequence. Defaults to 256.
            temperature (float, optional): Sampling temperature for diversity. Defaults to 0.7.
        """
        # Applying model’s native chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenizing the created chat template
        llm_input_dict = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            return_tensors="pt",
        )

        # Moving tensors to the model's actual device
        model_device = next(self.model.parameters()).device
        llm_input_dict = {k: v.to(model_device) for k, v in llm_input_dict.items()}

        # Generating the response in a separate thread
        generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(
                **llm_input_dict,
                streamer=self.streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            ),
        )
        generation_thread.start()

        # Streaming and yielding the generated tokens as they are being generated
        for generated_token in self.streamer:
            yield generated_token

        generation_thread.join()
