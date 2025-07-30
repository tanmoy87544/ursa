import os, sqlite3

from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from ursa.agents import ArxivAgent, ExecutionAgent, PlanningAgent, ResearchAgent
from langgraph.checkpoint.sqlite import SqliteSaver

header = """
Testing a HITL version of URSA. Direct a prompt to either the:
[Arxiver], [Executor], [Planner], [Researcher]

The agent will get your prompt, the output of the last agent (if any), and their previous history.
when done use the escape key [USER DONE].
"""
def main():
    workspace = "hitl_example"
    os.makedirs(workspace, exist_ok=True)
    edb_path = Path(workspace) / "executor_checkpoint.db"
    edb_path.parent.mkdir(parents=True, exist_ok=True)
    econn = sqlite3.connect(str(edb_path), check_same_thread=False)
    executor_checkpointer = SqliteSaver(econn)

    pdb_path = Path(workspace) / "planner_checkpoint.db"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pconn = sqlite3.connect(str(pdb_path), check_same_thread=False)
    planner_checkpointer = SqliteSaver(pconn)

    rdb_path = Path(workspace) / "researcher_checkpoint.db"
    rdb_path.parent.mkdir(parents=True, exist_ok=True)
    rconn = sqlite3.connect(str(rdb_path), check_same_thread=False)
    researcher_checkpointer = SqliteSaver(rconn)

    model = ChatLiteLLM(
        model="openai/o3",
        max_tokens=50000,
    )

    arxiv_agent = ArxivAgent(
        llm=model,
        summarize=True,
        process_images=False,
        max_results=10,
        database_path="database_hitl",
        summaries_path="database_summaries_hitl",
        vectorstore_path="vectorstores_hitl",
        download_papers=True,
    )
    executor   = ExecutionAgent(llm=model, checkpointer=executor_checkpointer)
    planner    = PlanningAgent(llm=model, checkpointer=planner_checkpointer)
    researcher = ResearchAgent(llm=model, checkpointer=researcher_checkpointer)

    executor_state   = None
    planner_state    = None
    researcher_state = None
    arxiv_state      = None

    done = False
    last_agent_result = ""

    print(header)
    while not done:
        user_prompt = input("User query: ")
        if "[USER DONE]" in user_prompt:
            done = True
            break

        if "[Arxiver]" in user_prompt:
            llm_search_query = model.invoke(f"The user stated {user_prompt}. Generate between 1 and 8 words for a search query to address the users need. Return only the words to search").content
            print("Searching ArXiv for ", llm_search_query)
            arxiv_result = arxiv_agent.run(arxiv_search_query=llm_search_query, context=last_agent_result + user_prompt)
            if arxiv_state:
                arxiv_state.append(arxiv_result)
            else:
                arxiv_state = [arxiv_result]
            last_agent_result = arxiv_result
            print(f"[ArXiv Agent Output]:\n {last_agent_result}")
            continue

        if "[Executor]" in user_prompt:
            if executor_state:
                executor_state["messages"].append(HumanMessage(f"The last agent output was: {last_agent_result}\n The user stated: {user_prompt}"))
                executor_state = executor.action.invoke(executor_state, {
                        "recursion_limit": 999999,
                        "configurable": {"thread_id": executor.thread_id},
                    })
                last_agent_result = executor_state["messages"][-1].content
            else:
                executor_state = {"workspace":workspace}
                executor_state["messages"] = [HumanMessage(f"The last agent output was: {last_agent_result}\n The user stated: {user_prompt}")]
                executor_state = executor.action.invoke(executor_state, {
                        "recursion_limit": 999999,
                        "configurable": {"thread_id": executor.thread_id},
                    })
                last_agent_result = executor_state["messages"][-1].content
            print(f"[Executor Agent Output]:\n {last_agent_result}")
            continue

        if "[Planner]" in user_prompt:
            if planner_state:
                planner_state["messages"].append(HumanMessage(f"The last agent output was: {last_agent_result}\n The user stated: {user_prompt}"))
                planner_state = planner.action.invoke(planner_state, {
                        "recursion_limit": 999999,
                        "configurable": {"thread_id": planner.thread_id},
                    })
                last_agent_result = planner_state["messages"][-1].content
            else:
                planner_state = {"messages":[HumanMessage(f"The last agent output was: {last_agent_result}\n The user stated: {user_prompt}")]}
                planner_state = planner.action.invoke(planner_state, {
                        "recursion_limit": 999999,
                        "configurable": {"thread_id": planner.thread_id},
                    })
                last_agent_result = planner_state["messages"][-1].content
            print(f"[Planner Agent Output]:\n {last_agent_result}")
            continue

        print("You did not invoke an agent or escape. What are you doing?")
    
if __name__ == "__main__":
    main()