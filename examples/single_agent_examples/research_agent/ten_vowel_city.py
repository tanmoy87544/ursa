import sys

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

from oppenai.agents import ResearchAgent


##### Run a simple example of a Reearch Agent.

# Define a simple problem
problem = "Find a city with as least 10 vowels in its name."

# Choose the LLM and 
model = ChatLiteLLM(
    model="openai/o3-mini",
    max_tokens=20000 
)

# Initialize the agent
researcher = ResearchAgent(llm=model)

# Solve the problem
research_output = researcher.run(problem)


# Print results
print("Final summary: \n", research_output["messages"][-1].content)
# for x in research_output["messages"]:
#     print(x.content)

print("Citations: \n", [x for x in research_output.get("urls_visited", None)])


