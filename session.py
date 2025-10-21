class ChatSession:
    def __init__(self, system_prompt: str = None):
        self.history: list[dict[str, str]] = []
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
