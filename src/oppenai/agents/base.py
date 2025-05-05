from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel


class BaseAgent:
    def __init__(self, llm: str | BaseChatModel, *args, **kwargs):
        match llm:
            case BaseChatModel():
                self.llm = llm

            case str():
                self.llm_provider, self.llm_model = llm.split("/")
                self.llm = ChatLiteLLM(
                    model=self.llm_model,
                    max_tokens=kwargs.get("max_tokens", 10000),
                    max_retries=kwargs.get("max_retries", 2),
                )
            case _:
                raise TypeError(
                    "llm argument must be a string with the provider and model, or a BaseChatModel instance."
                )
