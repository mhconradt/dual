## Setup

To run the app, you'll need to install its requirements and set the environment variables for OpenAI, Anthropic and Mistral:
```shell
pip install -r requirements.txt
export OPENAI_API_KEY=""
export ANTHROPIC_API_KEY=""
export MISTRAL_API_KEY=""
streamlit run Dual.py
```
It's recommended, but not strictly required to have API keys for all providers.