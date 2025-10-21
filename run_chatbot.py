import torch

from model_wrapper import ChatModel
from session import ChatSession


def main():
    chat = ChatSession(system_prompt="You are a helpful and concise assistant.")
    model = ChatModel(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    print(f"Chatbot '{model.model_name}' ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        chat.add_message(role="user", content=user_input)
        response = model.generate(messages=chat.get_history())

        chat.add_message(role="assistant", content=response)
        print(f"Assistant: {response}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
