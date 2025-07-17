import sys
import time
sys.path.append("../../.")

from ursa.agents                              import ArxivAgent
from langchain_openai                            import ChatOpenAI
from langchain_community.callbacks.openai_info   import OpenAICallbackHandler


def main():

    callback_handler = OpenAICallbackHandler()


    llm = ChatOpenAI(model      = "o3-mini",
                    max_tokens  = 10000,
                    timeout     = None,
                    max_retries =  2,
                    callbacks=[callback_handler],
                    )

    
    agent = ArxivAgent(llm=llm, summarize = True, process_images = False, 
                       max_results        = 100,   
                       database_path      ='database',
                       summaries_path     ='database_summaries', 
                       vectorstore_path   ='vectorstores', 
                       download_papers    = False)
    

    t0 = time.time()
    
    result = agent.run(arxiv_search_query="High Entropy Alloys", 
                       context="Find High entropy alloys suitable for application under extreme conditions")

    t1 = time.time()
    
    print(f"Time Taken: {t1-t0}")
    print(f"Total Tokens Used: {callback_handler.total_tokens}")
    print(f"Prompt Tokens: {callback_handler.prompt_tokens}")
    print(f"Completion Tokens: {callback_handler.completion_tokens}")
    print(f"Successful Requests: {callback_handler.successful_requests}")
    print(f"Total Cost (USD): ${callback_handler.total_cost}")


if __name__ == "__main__":
    main()
