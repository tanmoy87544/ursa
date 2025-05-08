import os
import feedparser
import requests
from urllib.parse import quote
from typing import TypedDict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from .base import BaseAgent

class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: str
    full_text: str

class PaperState(TypedDict, total=False):
    query: str
    papers: List[PaperMetadata]
    summaries: List[str]
    final_summary: str



class LiteratureAgent(BaseAgent):
    def __init__(self, llm = "OpenAI/o3-mini", max_results: int = 3, *args, **kwargs):
        super().__init__(llm, args, kwargs)
        self.max_results = max_results
        self.graph = self._build_graph()



    def _fetch_papers(self, query: str) -> List[PaperMetadata]:
        encoded_query = quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={self.max_results}"
        feed = feedparser.parse(url)

        papers = []
        for entry in feed.entries:
            arxiv_id = entry.id.split('/abs/')[-1]
            title = entry.title.strip()
            authors = ", ".join(author.name for author in entry.authors)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_filename = f"{arxiv_id}.pdf"

            try:
                response = requests.get(pdf_url)
                with open(pdf_filename, 'wb') as f:
                    f.write(response.content)

                loader = PyPDFLoader(pdf_filename)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])[:]  # Consider setting limit to the number of words
            except Exception as e:
                full_text = f"Error fetching paper: {e}"
            finally:
                if os.path.exists(pdf_filename):
                    os.remove(pdf_filename)

            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "full_text": full_text,
            })

        return papers


    def _fetch_node(self, state: PaperState) -> PaperState:
        papers = self._fetch_papers(state["query"])
        return {**state, "papers": papers}

    
    def _summarize_node(self, state: PaperState) -> PaperState:
        summaries = []
        prompt = ChatPromptTemplate.from_template("""
You are a scientific assistant. Summarize the key contributions of this paper:

{paper}
""")
        chain = prompt | self.llm | StrOutputParser()

        for paper in state["papers"]:
            summary = chain.invoke({"paper": paper["full_text"]})
            summaries.append(summary)

        return {**state, "summaries": summaries}



    def _aggregate_node(self, state: PaperState) -> PaperState:
        summaries = state["summaries"]
        papers = state["papers"]
        formatted = []

        for i, (paper, summary) in enumerate(zip(papers, summaries)):
            citation = f"[{i+1}] {paper['title']} by {paper['authors']}\nLink: https://arxiv.org/abs/{paper['arxiv_id']}"
            formatted.append(f"{citation}\n\nSummary:\n{summary}")

        combined = "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(formatted)
        return {**state, "final_summary": combined}


    def _build_graph(self):
        builder = StateGraph(PaperState)

        builder.add_node("fetch_papers", self._fetch_node)
        builder.add_node("summarize_each", self._summarize_node)
        builder.add_node("aggregate", self._aggregate_node)
        
        builder.set_entry_point("fetch_papers")
        builder.add_edge("fetch_papers", "summarize_each")
        builder.add_edge("summarize_each", "aggregate")
        builder.set_finish_point("aggregate")
        
        return builder.compile()

    
    def run(self, query: str) -> str:
        result = self.graph.invoke({"query": query})
        return result.get("final_summary", "No summary generated.")



if __name__ == "__main__":

    agent = LiteratureAgent()

    result = agent.run("Experimental Constraints on neutron star radius")
        
    print(result)
        