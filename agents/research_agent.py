from langchain_core.messages      import HumanMessage, SystemMessage
from langgraph.prebuilt           import create_react_agent
from langchain_core.tools         import tool

from langgraph.graph              import END, StateGraph, START
from langgraph.graph.message      import add_messages
from typing_extensions            import TypedDict

from langchain_community.tools    import DuckDuckGoSearchResults
from langchain_community.tools    import TavilySearchResults

from typing   import Annotated, Optional, List
from pydantic import Field
from bs4      import BeautifulSoup

import requests

from .base                             import BaseAgent
from ..prompt_library.research_prompts import research_prompt, reflection_prompt

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    urls_visited: Optional[List[str]]
    max_research_steps: Optional[int] =  Field(default=100, description="Maximum number of research steps")

class ResearchAgent(BaseAgent):
    def __init__(self, llm = "OpenAI/gpt-4o", *args, **kwargs):
        super().__init__(llm, args, kwargs)
        self.research_prompt    = research_prompt
        self.reflection_prompt  = reflection_prompt
        self.tools              = [search_tool, process_content]
        self._initialize_agent()

    def review_node(self, state: ResearchState) -> ResearchState:
        translated = [SystemMessage(content=reflection_prompt)] + state["messages"]
        res        = self.llm.invoke(translated)
        print("Review Node: ", res.content)
        return {"messages": [HumanMessage(content=res.content)]}

    def response_node(self, state: ResearchState) -> ResearchState:
        messages = state["messages"]
        print("Response Node: ", messages[-2])
        return {"messages": [messages[-2]]}

    def _initialize_agent(self):
        self.graph = StateGraph(ResearchState)
        self.graph.add_node("research", create_react_agent(self.llm, self.tools, prompt=self.research_prompt))

        self.graph.add_node("review",     self.review_node)
        self.graph.add_node("response", self.response_node)

        self.graph.add_edge(START,       "research")
        self.graph.add_edge("research",    "review")
        self.graph.add_edge("response",         END)

        self.graph.add_conditional_edges(
            "review", 
            should_continue,
            {
                "research":"research",
                "response":"response"
            }
        )
        self.action = self.graph.compile()



@tool
def process_content(url: str) -> str:
    """
    Processes content from a given webpage.
    
    Args:
        url: string with the url to obtain text content from
    """
    print("Parsing information from ", url)
    response = requests.get(url)
    soup     = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced",include_answer=True)

# tool_node = ToolNode(tools)
# llm.bind_tools(tools)

def should_continue(state: ResearchState):
    if len(state["messages"]) > (state.get("max_research_steps",100)+3):
        return "response"
    if "[APPROVED]" in state["messages"][-1].content:
        return "response"
    return "research"

def main():
    researcher = ResearchAgent()
    problem_string = "Who are the 2025 Detroit Tigers top 10 prospects and what year were they born?" 
    inputs = {"messages": [HumanMessage(content=problem_string)]}
    result = researcher.action.invoke(inputs)
    for x in result["messages"]:
        print(x.content)
    return result

if __name__ == "__main__":
    main()

