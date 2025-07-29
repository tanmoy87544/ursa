# URSA - The LANL Scientific Agent

<img src="./logos/logo.png" alt="URSA Logo" width="200" height="200">

The flexible agentic workflow for accelerating scientific tasks. 
Composes information flow between agents for planning, code writing and execution, and online research to solve complex problems.

## Installation
With pip:
```console
pip install git+ssh://git@lisdi-git.lanl.gov:10022/artimis/ursa.git
```

With uv:
```console
uv add git+ssh://git@lisdi-git.lanl.gov:10022/artimis/ursa.git
```

## How to use this code
Better documentation will be incoming, but for now there are examples in the examples folder that should give
a decent idea for how to set up some basic problems. They also should give some idea of how to pass results from
one agent to another. I will look to add things with multi-agent graphs, etc. in the future. 

Documentation for each URSA agent:
- [Planning Agent](docs/planning_agent.md)
- [Execution Agent](docs/execution_agent.md)
- [ArXiv Agent](docs/arxiv_agent.md)
- [Research Agent](docs/research_agent.md)
- [Hypothesizer Agent](docs/hypothesizer_agent.md)

Documentation for combining agents:
- [ArXiv -> Execution for Materials](docs/combining_arxiv_and_execution.md )
- [ArXiv -> Execution for Neurton Star Properties](docs/combining_arxiv_and_execution_neutronStar.md  )

# Sandboxing
The Execution Agent is allowed to run system commands and write/run code. Being able to execute arbitrary system commands or write
and execute code has the potential to cause problems like:
- Damage LANL code or data on the computer
- Damage the compter
- Transmit LANL data

The Research Agent scrapes data from urls, so has the potential to attempt to pull information from questionable sources.

Some suggestions for sandboxing the agent:
- Creating a specific environment such that limits URSA's access to only what you want. Examples:
    - Creating/using a virtual machine that is sandboxed from the rest of your machine
    - Creating a new account on your machine specifically for URSA 
- Creating a network blacklist/whitelist to ensure that network commands and webscraping are contained to safe sources

You have a duty for ensuring that you use URSA responsibly.

