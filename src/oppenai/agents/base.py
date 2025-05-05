from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel


class BaseAgent:
    # llm: BaseChatModel
    # llm_with_tools: Runnable[LanguageModelInput, BaseMessage]

    def __init__(self, llm: str | BaseChatModel, **kwargs):
        match llm:
            case BaseChatModel():
                self.llm = llm

            case str():
                self.llm_provider, self.llm_model = llm.split("/")
                self.llm = ChatLiteLLM(
                    model=self.llm_model,
                    max_tokens=kwargs.pop("max_tokens", 10000),
                    max_retries=kwargs.pop("max_retries", 2),
                    **kwargs,
                )

            case _:
                raise TypeError(
                    "llm argument must be a string with the provider and model, or a BaseChatModel instance."
                )
