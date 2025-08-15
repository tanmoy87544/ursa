import sys

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from ursa.agents import ExecutionAgent, MaterialsProjectAgent

import json, os
# make sure your MP_API_KEY is set in env or pass it here
agent = MaterialsProjectAgent(max_results=5)
# example query: 5 stable Ga/In semiconductors with 1–2 eV gaps
query = {
    "elements":      ["Ga", "In"],
    "band_gap_min":  1.0,
    "band_gap_max":  2.0
}
context = (
    "1) List each material’s band gap and stability.\n"
    "2) Then, for the single lowest-energy-above-hull candidate, "
    "retrieve and explain its crystal structure."
)
print(agent.run(mp_query=query, context=context))
