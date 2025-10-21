# Conversational Chatbot

This is a repository dedicated to my attempt to create an interface for interacting with a LLM. Currently it represents a simple script with infinite loop for the continuity of the conversation but I plan to add more features:

* Saving and loading of conversations.
* Chatbot as a Streamlit app.
* Chatbot as a FastAPI endpoint.

## Launching chatbot

Firstly, one should clone the repository:

```bash
git clone https://github.com/spolivin/conv-chatbot

cd conv-chatbot
```

Secondly, create a python virtual environment to install the dependencies or use conda for simplicity like so:

```bash
conda create -y -n chat-venv python=3.10
conda activate chat-venv
pip install -r requirements.txt
```

Now, we can run the chatbot:

```bash
python run_chatbot.py
```
> Currently the script uses `Qwen/Qwen2.5-1.5B-Instruct` model from Hugging Face. The model has been chosen due to its relatively small size and due to the absence of GPU on a local machine. Despite these limitations, the script is adapted to efficiently run the model on both CPU and GPU.

For instance, an excerpt from the possible conversation:

```
You: Hello!
Assistant: Hello! How can I assist you today?

You: What is meant by a proton? 
Assistant: A proton is a subatomic particle that has a positive electric charge, with an atomic number of 1 and mass equal to approximately 1.6726 × 10^-27 kilograms.

Protons are found in the nucleus of every atom and play a crucial role in determining the chemical properties of elements. They combine with neutrons and electrons to form atoms and molecules. Protons have an important effect on the stability of the nucleus and contribute to the overall structure of the atom.
```
> Enter `quit` to exit the chatbot.

Reponses from the user and assistant are saved and appended to chat history after each turn, thus enabling the LLM to remember what has been said previously and keeping the conversation going. Furthermore, using chat templates helps the model to adequately generate its responses in ChatGPT-like fashion.
