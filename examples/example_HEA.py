import sys
sys.path.append("../../.")

from lanl_scientific_agent.agents import ArxivAgent


def main():
    agent = ArxivAgent(llm="OpenAI/o3-mini", summarize = True, process_images = False, 
                       max_results = 100,   
                       database_dir='database',
                       download_papers=False)
    
    result = agent.run(arxiv_search_query="High Entropy Alloys", 
                       context="Find High entropy alloys suitable for application under extreme conditions")
    
    print(result)


if __name__ == "__main__":
    main()
