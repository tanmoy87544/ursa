import sys

from oppenai.agents                              import ArxivAgent
from langchain_openai                            import ChatOpenAI

def main():

    llm = ChatOpenAI(model      = "o3-mini",
                    max_tokens  = 10000,
                    timeout     = None,
                    max_retries =  2,
                    )


    agent = ArxivAgent(llm=llm, summarize = True, process_images = True, 
                       max_results        = 3,   
                       database_path      ='database_neutron_star',
                       summaries_path     ='database_summaries_neutron_star', 
                       vectorstore_path   ='vectorstores_neutron_star', 
                       download_papers    = True)

    
    result = agent.run(arxiv_search_query="Experimental Constraints on neutron star radius", 
                       context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?")
    
    print(result)


if __name__ == "__main__":
    main()
