import json
import os
import uuid


class ChatSession:
    def __init__(self, system_prompt: str = None):
        # Chat history

        self.history: list[dict[str, str]] = []

        # Conversation identifier
        self.chat_id: str = uuid.uuid4().hex[:6]

        # Saved chats folder
        self.save_dir = "chats"

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def add_message(self, role: str, content: str):
        """Adds a message to the chat history.

        Args:
            role (str): User or assistant.
            content (str): Message text.
        """
        self.history.append({"role": role, "content": content})

    def get_history(self) -> list[dict[str, str]]:
        """Retrieves the entire chat history.

        Returns:
            list[dict[str, str]]: Chat history.
        """
        return self.history

    def save_history(self):
        """Saves chat history as JSON.

        Args:
            save_dir (str, optional): Folder to store saved chats. Defaults to "chats".
        """
        os.makedirs(self.save_dir, exist_ok=True)
        with open(f"{self.save_dir}/{self.chat_id}.json", "w") as json_file:
            json.dump(self.get_history(), json_file, indent=4)

    def load_history(self, chat_id: str):
        """Loads chat history from JSON.

        Args:
            chat_id (str): Conversation ID.
            save_dir (str, optional): Folder from which to load chat. Defaults to "chats".
        """
        self.chat_id = chat_id
        with open(f"{self.save_dir}/{chat_id}.json", "r") as json_file:
            self.history = json.load(json_file)
