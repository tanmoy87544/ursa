import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import LiteratureAgent


def main():
    agent = LiteratureAgent()
    result = agent.run("Experimental Constraints on neutron star radius")
    print(result)


if __name__ == "__main__":
    main()
