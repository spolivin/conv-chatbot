import argparse
import sys

import torch

from chatbot_api import ChatModel, ChatSession

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name",
    type=str,
    help="Model name for assistant",
    default="Qwen/Qwen2.5-1.5B-Instruct",
)
parser.add_argument(
    "--sys-prompt",
    type=str,
    help="System prompt",
    default="You are a helpful and concise assistant.",
)
parser.add_argument("--max-new-tokens", type=int, help="Max new tokens", default=2048)
parser.add_argument("--temperature", type=float, help="Temperature", default=0.8)
parser.add_argument("--chat-id", type=str, help="Conversation ID", default=None)
args = parser.parse_args()


def display_chat_history(history: list[dict[str, str]]):
    """Displays all messages belonging to the chat.

    Args:
        history (list[dict[str, str]]): Chat history.
    """
    for turn in history:
        message = turn["content"]
        if turn["role"] == "user":
            print(f"You: {message}")
        elif turn["role"] == "assistant":
            print(f"Assistant: {message}\n")
        else:
            pass


def main():
    chat = ChatSession(system_prompt=args.sys_prompt)
    model = ChatModel(model_name=args.model_name)

    # Loading a specific chat history (if provided in args)
    if args.chat_id is not None:
        try:
            chat.load_history(chat_id=args.chat_id)
            display_chat_history(history=chat.get_history())
        except FileNotFoundError:
            print(f"Chat #{args.chat_id} is not found. Aborting chatbot.")
            sys.exit(1)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            # Saving chat history
            chat.save_history()
            print("\nHave a nice day!")
            break

        chat.add_message(role="user", content=user_input)

        print("Assistant: ", end="")
        # Streaming response from the model
        response = model.generate(
            messages=chat.get_history(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        chat.add_message(role="assistant", content=response)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
