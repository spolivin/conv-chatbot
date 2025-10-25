import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatModel:
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: torch.device = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    ) -> str:
        """Generates a LLM response.

        Args:
            messages (list[dict[str, str]]): Chat history.
            max_new_tokens (int, optional): Maximum number of tokens to generate in the output sequence. Defaults to 256.
            temperature (float, optional): Sampling temperature for diversity. Defaults to 0.7.

        Returns:
            str: LLM response.
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

        # Generating the response
        with torch.no_grad():
            output_ids = self.model.generate(
                **llm_input_dict,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # Decoding only the newly generated tokens
        input_len = llm_input_dict["input_ids"].shape[-1]
        generated_text = self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        return generated_text.strip()
