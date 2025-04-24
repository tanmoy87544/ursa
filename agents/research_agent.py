from langchain_core.messages      import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt           import create_react_agent
from langchain_core.tools         import tool

from langgraph.graph              import END, StateGraph, START
from langgraph.graph.message      import add_messages
from typing_extensions            import TypedDict

from langchain_community.tools    import DuckDuckGoSearchResults
from langchain_community.tools    import TavilySearchResults
# from langchain_core.runnables.graph import MermaidDrawMethod

from typing   import Annotated, Optional, List
from pydantic import Field
from bs4      import BeautifulSoup

import requests
import inspect
# import primp
# client = primp.Client(verify=True, ca_cert_file="/Users/mikegros/Downloads/ZS_Root_CA.pem")

from .base                              import BaseAgent
from ..prompt_library.research_prompts  import research_prompt, reflection_prompt, summarize_prompt
from ..util.tool_helper                 import class_bound_tool, postprocess_tools
# --- ANSI color codes ---
BLUE  = "\033[1;34m"
RED   = "\033[1;31m"
GREEN = "\033[92m"
RESET = "\033[0m"

class ResearchState(TypedDict):
    research_query: str
    messages: Annotated[list, add_messages]
    urls_visited: List[str]
    max_research_steps: Optional[int] =  Field(default=100, description="Maximum number of research steps")

class ResearchAgent(BaseAgent):
    def __init__(self, llm = "OpenAI/gpt-4o", *args, **kwargs):
        super().__init__(llm, args, kwargs)
        self.research_prompt    = research_prompt
        self.reflection_prompt  = reflection_prompt
        # cb_tools                = postprocess_tools([self.process_content])
        # print(cb_tools[0].func(url="www.google.com"))
        self.tools              = [search_tool, process_content] # + cb_tools
        self._initialize_agent()

    def review_node(self, state: ResearchState) -> ResearchState:
        for x in state["messages"][-3:]:
            print("MESSAGES: ",x.content)
        translated = [SystemMessage(content=reflection_prompt)] + state["messages"]
        res        = self.llm.invoke(translated)
        return {"messages": [HumanMessage(content=res.content)]}

    def response_node(self, state: ResearchState) -> ResearchState:
        messages = state["messages"] + [SystemMessage(content=summarize_prompt)]
        response = self.llm.invoke(messages)
        
        urls_visited = []
        for message in messages:
            if message.model_dump().get("tool_calls",[]):
                if "url" in message.tool_calls[0]["args"]:
                    urls_visited.append(message.tool_calls[0]["args"]["url"])
        return {"messages": [response.content], "urls_visited":urls_visited}

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
        # self.action.get_graph().draw_mermaid_png(output_file_path="./research_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

def process_content(url: str) -> str: #, context: str) -> str:
    """
    Processes content from a given webpage.
    
    Args:
        url: string with the url to obtain text content from
    """
    print("Parsing information from ", url)
    response = requests.get(url)
    soup     = BeautifulSoup(response.content, 'html.parser')

    #summarized_information = self.llm.invoke()
    return soup.get_text()

search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced",include_answer=True)


def should_continue(state: ResearchState):
    if len(state["messages"]) > (state.get("max_research_steps",100)+3):
        return "response"
    if "[APPROVED]" in state["messages"][-1].content:
        return "response"
    return "research"

def main():
    researcher = ResearchAgent(llm="OpenAI/gpt-4o")
    problem_string = "Who are the 2025 Detroit Tigers top 10 prospects and what year were they born?" 
    inputs = {"messages": [HumanMessage(content=problem_string)]}
    result = researcher.action.invoke(inputs, {'recursion_limit':10000})
    
    colors = [BLUE, RED]
    for ii,x in enumerate(result["messages"][:-1]):
        if type(x) != ToolMessage:
            print(f"{colors[ii % 2]}" + x.content+f"{RESET}")
    print(80*"#")
    print(f"{GREEN}"+result["messages"][-1].content+f"{RESET}")
    print("Citations: ", result["urls_visited"])
    return result

if __name__ == "__main__":
    main()

