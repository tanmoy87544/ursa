import os
from .base import BaseAgent
from ..util.memory_logger import AgentMemory


class RecallAgent(BaseAgent):
    def __init__(self, llm, embedding, **kwargs):
        
        super().__init__(llm, **kwargs)
        self.memorydb = AgentMemory(embedding_model=embedding)
    
    def remember(self, query):
        memories = self.memorydb.retrieve(query)
        summarize_query = f"""
        You are being given the critical task of generating a detailed description of logged information 
        to an important official to make a decision. Summarize the following memories that are related to 
        the statement. Ensure that any specific details that are important are retained in the summary.

        Query: {query}

        """

        for memory in memories:
            summarize_query += f"Memory: {memory} \n\n"
        memory = self.llm.invoke(summarize_query).content
        return memory
