from langchain_openai             import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama


class BaseAgent:
    def __init__(self, llm, *args, **kwargs):
        if type(llm) == str:
            self.llm_provider, self.llm_model = llm.split(":")
            if self.llm_provider.lower() == "openai":
                self.llm = ChatOpenAI(
                    model       = self.llm_model,
                    max_tokens  = kwargs.get("max_tokens",10000),
                    timeout     = kwargs.get("timeout",    None),
                    max_retries = kwargs.get("max_retries",   2)
                )
            elif self.llm_provider.lower() == "ollama":
                self.llm = ChatOllama(
                    model       = self.llm_model,
                    max_tokens  = kwargs.get("max_tokens",10000)
                )
            else:
                raise TypeError("llm argument must be a string with the provider and model or a model itself")
        else:
            self.llm = llm
                