import sys

from oppenai.agents import ArxivAgent


def main():
    agent = ArxivAgent()
    result = agent.run(arxiv_search_query="Experimental Constraints on neutron star radius", 
                       context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?")
    print(result)


if __name__ == "__main__":
    main()
