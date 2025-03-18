from langchain_core.messages      import HumanMessage
from langchain_core.prompts       import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.tools         import tool
from langgraph.prebuilt           import ToolNode
from langchain_ollama.chat_models import ChatOllama
from langchain_openai             import ChatOpenAI
from langchain_core.messages      import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts       import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph              import END, StateGraph, START
from langgraph.graph.message      import add_messages
from langgraph.checkpoint.memory  import MemorySaver
from typing_extensions            import TypedDict

from typing   import Annotated, List, Sequence, Dict, Any
from pydantic import BaseModel, Field

from langchain_community.tools import TavilySearchResults

search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True)

from parse import extract_json
import asyncio

# llm = ChatOllama(
#     model       = "gemma3:12b",
#     # model       = "phi4",
#     max_tokens  = 4000,
#     timeout     = None,
#     max_retries = 2
# )

llm = ChatOpenAI(
    model       = "o3-mini",
    max_tokens  = 10000,
    timeout     = None,
    max_retries = 2
)

# llm.bind_tools([search])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
        """
        You are planning a scientific investigation to solve the given problem:
        
        Please create a step-by-step plan to solve this problem. 
        Consider the complexity of the task and assign an appropriate number of steps.
        Each step should be a well-defined task that can be implemented and evaluated.
        For each step, specify:
        
        1. A descriptive name for the step
        2. A detailed description of what needs to be done
        3. Whether the step requires generating and executing code
        4. Expected outputs of the step
        5. How to evaluate whether the step was successful
        
        Consider a diverse range of appropriate steps such as:
        - Data gathering or generation
        - Data preprocessing and cleaning
        - Analysis and modeling
        - Hypothesis testing
        - Visualization
        - Evaluation and validation

        Only allocate the steps that are needed to solve the problem.        
        """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

formalize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
        """
        Now that the step-by-step plan is finalized, format it into a series of steps in the form of a JSON array with objects having the following structure:
        [
            {{
                "id": "unique_identifier",
                "name": "Step name",
                "description": "Detailed description of the step",
                "requires_code": true/false,
                "expected_outputs": ["Output 1", "Output 2", ...],
                "success_criteria": ["Criterion 1", "Criterion 2", ...]
            }},
            ...
        ]
        """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a critical reviewer being given a series of steps to solve a problem."
            "Provide detailed recommendations, including missing or superfluous steps. In the end, decide if the current proposal should be approved or revised."
            "Ensure the proposed effort is appropriate for the problem.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def printpipe(x):
    print("="*50)
    print(x)
    print("&"*50)
    return x

generate  = prompt            | llm
reflect   = reflection_prompt | llm
formalize = formalize_prompt  | llm

class PlanningState(TypedDict):
    messages: Annotated[list, add_messages]
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Ordered steps in the solution plan")
    
def generation_node(state: PlanningState) -> PlanningState:
    return {"messages": [generate.invoke(state["messages"])]}

def formalize_node(state: PlanningState) -> PlanningState:
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    for _ in range(10):
        try:
            res      = formalize.invoke(translated)
            json_out = extract_json(res.content)
            break
        except ValueError:
            translated.append(HumanMessage(content="Your response was not valid JSON. Try again."))
    print("Formalized", res.content)

    return {"messages": [HumanMessage(content=res.content)], "plan_steps": json_out}    

def reflection_node(state: PlanningState) -> PlanningState:
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = reflect.invoke(translated)

    return {"messages": [HumanMessage(content=res.content)]}

def should_continue(state: PlanningState):
    if len(state["messages"]) > 6:
        # End after 3 iterations
        return "formalize"
    return "reflect"

# tools       = [search]
# search_node = ToolNode(tools)

builder = StateGraph(PlanningState)
builder.add_node("generate", generation_node)
# builder.add_node("search",       search_node)
builder.add_node("reflect",  reflection_node)
builder.add_node("formalize", formalize_node)

builder.add_edge(START,       "generate")
builder.add_edge("reflect",   "generate")
builder.add_edge("formalize",        END)

# builder.add_conditional_edges(
#     "generate",
#     should_search,
#     {
#         "search": "search",
#         "reflect": "reflect"
#     },
# )

builder.add_conditional_edges(
    "generate", 
    should_continue,
    {
        "reflect":"reflect",
        "formalize":"formalize"
    }
    )

memory = MemorySaver()
# graph = builder.compile()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def main():
    for event in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Find a city with as least 10 vowels in its name." #"Write an essay on ideal high-entropy alloys for spacecraft."
                )
            ],
        },
        config,
    ):
        print(event)
        print("---")

# asyncio.run()
if __name__ == "__main__":
    main()