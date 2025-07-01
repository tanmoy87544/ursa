import os
import pymupdf 
import requests
import feedparser
from PIL import Image
from io import BytesIO
import base64
from urllib.parse import quote
from typing_extensions import TypedDict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
# from langchain_core.runnables.graph import MermaidDrawMethod

from openai import OpenAI

from .base import BaseAgent

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# === OpenAI Vision Client ===
client = OpenAI()

# === Data Schemas ===
class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: str
    full_text: str

class PaperState(TypedDict, total=False):
    query: str
    context: str
    papers: List[PaperMetadata]
    summaries: List[str]
    final_summary: str


# === Image Description with GPT-4V ===
def describe_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are a scientific assistant who explains plots and scientific diagrams."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this scientific image or plot in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ],
            },
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def extract_and_describe_images(pdf_path: str, max_images: int = 5) -> List[str]:
    doc = pymupdf.open(pdf_path)
    descriptions = []
    image_count = 0

    for page_index in range(len(doc)):
        if image_count >= max_images:
            break
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            if image_count >= max_images:
                break
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))

            try:
                desc = describe_image(image)
                descriptions.append(f"Page {page_index + 1}, Image {img_index + 1}: {desc}")
            except Exception as e:
                descriptions.append(f"Page {page_index + 1}, Image {img_index + 1}: [Error: {e}]")
            image_count += 1

    return descriptions


# === Main Agent ===
class ArxivAgent(BaseAgent):
    def __init__(self, llm="openai/o3-mini", process_images = True, max_results: int = 3, *args, **kwargs):
        super().__init__(llm, **kwargs)
        self.max_results = max_results
        self.process_images = process_images
        self.graph = self._build_graph()

    def _fetch_papers(self, query: str) -> List[PaperMetadata]:
        print(f"{BOLD}{BLUE}ArXiv Agent beginning workflow{RESET}")
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
                full_text = "\n".join([p.page_content for p in pages])

                if self.process_images:
                    image_descriptions = extract_and_describe_images(pdf_filename)
                    full_text += "\n\n[Image Interpretations]\n" + "\n".join(image_descriptions)

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
        if self.process_images:
            prompt = ChatPromptTemplate.from_template("""
            You are a scientific assistant helping summarize research papers.
            
            The paper below consists of:
            - Main written content (from the body of the PDF)
            - Descriptions of images and plots extracted via visual analysis (clearly marked at the end)
            
            Your task is to summarize the paper in the following context: {context}
            
            in two separate sections:
            
            1. **Text-Based Insights**: Summarize the main contributions and findings from the written text.
            2. **Image-Based Insights**: Describe what the extracted image/plot interpretations add or illustrate. If the image data supports or contradicts the text, mention that.
            
            Here is the paper content:
            
            {paper}
            """)
        else:
            prompt = ChatPromptTemplate.from_template("""
            You are a scientific assistant helping summarize research papers.
            
            The paper below consists of the main written content (from the body of the PDF)
            
            Your task is to summarize the paper in the following context: {context}
            
            Here is the paper content:
            
            {paper}
            """)
        chain = prompt | self.llm | StrOutputParser()
    
        for paper in state["papers"]:
            print(f"{BOLD}Summarizing paper: {RED}: {paper['arxiv_id']} - {paper['title']} {RESET}" )
            summary = chain.invoke({"paper": paper["full_text"], "context":state["context"]}, {"configurable": {"thread_id": self.thread_id}})
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
        
        graph = builder.compile(checkpointer=self.checkpointer)
        # graph.get_graph().draw_mermaid_png(output_file_path="arxiv_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)
        return graph

    def run(self, arxiv_search_query: str, context: str, recursion_limit=100) -> str:
        result = self.graph.invoke({"query": arxiv_search_query, "context":context}, {"recursion_limit":recursion_limit, "configurable": {"thread_id": self.thread_id}})
        return result.get("final_summary", "No summary generated.")



if __name__ == "__main__":
    agent = ArxivAgent()
    result = agent.run(arxiv_search_query="Experimental Constraints on neutron star radius", 
                       context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?")
    print(result)


