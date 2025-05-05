import os
from typing import Annotated

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from ..prompt_library.literature_prompts import search_prompt, summarize_prompt
from .base import BaseAgent

workspace_dir = "./workspace/"
os.makedirs(workspace_dir, exist_ok=True)


class LiteratureState(TypedDict):
    messages: Annotated[list, add_messages]


class LiteratureAgent(BaseAgent):
    def __init__(
        self, llm: str | BaseChatModel = "openai/gpt-4o-mini", **kwargs
    ):
        super().__init__(llm, **kwargs)
        self.search_prompt = search_prompt
        self.summarize_prompt = summarize_prompt
        self.tools = [arxiv_tool]
        self.tool_node = ToolNode(self.tools)
        self.llm = self.llm.bind_tools(self.tools)

        self._initialize_agent()

    # Define the function that calls the model
    def query_executor(self, state: LiteratureState) -> LiteratureState:
        messages = state["messages"]
        if type(state["messages"][0]) == SystemMessage:
            state["messages"][0] = SystemMessage(content=self.search_prompt)
        else:
            state["messages"] = [
                SystemMessage(content=self.search_prompt)
            ] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    # Define the function that calls the model
    def summarize(self, state: LiteratureState) -> LiteratureState:
        messages = [SystemMessage(content=self.summarize_prompt)] + state[
            "messages"
        ]
        response = self.llm.invoke(messages)
        return {"messages": [response.content]}

    def _initialize_agent(self):
        self.graph = StateGraph(LiteratureState)

        self.graph.add_node("agent", self.query_executor)
        self.graph.add_node("action", self.tool_node)
        self.graph.add_node("summarize", self.summarize)

        self.graph.add_edge(START, "agent")
        self.graph.add_edge("agent", "action")
        self.graph.add_edge("action", "summarize")
        self.graph.add_edge("summarize", END)

        self.action = self.graph.compile()


arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, load_all_available_meta=True)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)


def main():
    literature_agent = LiteratureAgent()
    problem_string = "Check if a neutron star radius of 5 km is consistent with the literature."
    inputs = {"messages": [HumanMessage(content=problem_string)]}
    result = literature_agent.action.invoke(inputs)
    print(result["messages"][-1].content)
    return result


if __name__ == "__main__":
    main()
