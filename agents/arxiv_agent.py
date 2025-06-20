import os
import pymupdf  # PyMuPDF
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
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
# from langchain_core.runnables.graph import MermaidDrawMethod

from openai import OpenAI

from .base import BaseAgent


# === OpenAI Vision Client ===
client = OpenAI()

# === Data Schemas ===
class PaperMetadata(TypedDict):
    arxiv_id: str
    #title: str
    #authors: str
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
    def __init__(self, llm="OpenAI/o3-mini", summarize: bool = True, process_images = True, max_results: int = 3,
                 database_dir='database', download_papers: bool = True, *args, **kwargs):
        super().__init__(llm, args, kwargs)
        self.summarize = summarize
        self.process_images = process_images
        self.max_results = max_results
        self.database_dir = database_dir
        self.download_papers = download_papers
        
        self.graph = self._build_graph()

        os.makedirs(self.database_dir, exist_ok=True)

        os.makedirs('database_summaries', exist_ok=True)

    def _fetch_papers(self, query: str) -> List[PaperMetadata]:
    
        if self.download_papers:
            
            encoded_query = quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={self.max_results}"
            feed = feedparser.parse(url)
    
            for i,entry in enumerate(feed.entries):
                full_id = entry.id.split('/abs/')[-1]
                arxiv_id = full_id.split('/')[-1]
                title = entry.title.strip()
                authors = ", ".join(author.name for author in entry.authors)
                pdf_url = f"https://arxiv.org/pdf/{full_id}.pdf"
                pdf_filename = os.path.join(self.database_dir, f"{arxiv_id}.pdf")
    
                if os.path.exists(pdf_filename):
                    print(f"Paper # {i+1}, Title: {title}, already exists in database")
                else:
                    print(f"Downloading paper # {i+1}, Title: {title}")
                    response = requests.get(pdf_url)
                    with open(pdf_filename, 'wb') as f:
                        f.write(response.content)
                        

        papers = []
        for i,pdf_filename in enumerate(os.listdir(self.database_dir)):
            full_text = ""
            if self.summarize:
            #if False:
                try:
                    loader = PyPDFLoader( os.path.join(self.database_dir, pdf_filename) )
                    pages = loader.load()
                    full_text = "\n".join([p.page_content for p in pages])
        
                    if self.process_images:
                        image_descriptions = extract_and_describe_images(pdf_filename)
                        full_text += "\n\n[Image Interpretations]\n" + "\n".join(image_descriptions)
                        
                except Exception as e:
                    full_text = f"Error loading paper: {e}"
    
            papers.append({
                "arxiv_id": pdf_filename.split('.pdf')[0],
                #"title": title,
                #"authors": authors,
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
            You are a scientific assistant helping extract insights from research papers.
            
            The paper below consists of the main written content (from the body of the PDF)
            
            Your task is to read the paper and provide a short summary that accomplishes the task: {context}

            If the paper is irrelevant to the task, then just state it as such.
            
            Here is the paper content:
            
            {paper}
            """)
        chain = prompt | self.llm | StrOutputParser()
    
        for paper in state["papers"]:
            arxiv_id = paper["arxiv_id"]
            summary_filename = os.path.join('database_summaries', f"{arxiv_id}_summary.txt")
            if os.path.exists(summary_filename):
                with open(summary_filename, 'r') as f:
                    summaries.append(f.read())

            else:
                try:
                    summary = chain.invoke({"paper": paper["full_text"], "context":state["context"]})
                    
                except Exception as e:
                    summary = f"Error summarizing paper: {e}"
                    
                summaries.append(summary)
                
                with open(summary_filename, "w") as f:
                    f.write(summary)
    
        return {**state, "summaries": summaries}


    
    def _aggregate_node(self, state: PaperState) -> PaperState:
        summaries = state["summaries"]
        papers = state["papers"]
        formatted = []

        for i, (paper, summary) in enumerate(zip(papers, summaries)):
            citation = f"[{i+1}] Arxiv ID: {paper['arxiv_id']}"
            formatted.append(f"{citation}\n\nSummary:\n{summary}")

        combined = "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(formatted)

        with open('summaries_combined.txt', "w") as f:
            f.write(combined)


        prompt = ChatPromptTemplate.from_template("""
            You are a scientific assistant helping extract insights from summaries of research papers.
            
            Here are the summaries of a large number of scientific papers:

            {Summaries}
            
            Your task is to read all the summaries and provide a response to this task: {context}
            """)

        chain = prompt | self.llm | StrOutputParser()

        final_summary = chain.invoke({"Summaries": combined, "context":state["context"]})

        with open('final_summary.txt', "w") as f:
            f.write(final_summary)

        return {**state, "final_summary": final_summary}


    
    def _build_graph(self):
        builder = StateGraph(PaperState)
        builder.add_node("fetch_papers", self._fetch_node)

        if self.summarize:
            builder.add_node("summarize_each", self._summarize_node)
            builder.add_node("aggregate", self._aggregate_node)

            builder.set_entry_point("fetch_papers")
            builder.add_edge("fetch_papers", "summarize_each")
            builder.add_edge("summarize_each", "aggregate")
            builder.set_finish_point("aggregate")
            #builder.set_finish_point("summarize_each")

        else:
            builder.set_entry_point("fetch_papers")
            builder.set_finish_point("fetch_papers")
    

        graph = builder.compile()
        return graph

    def run(self, arxiv_search_query: str, context: str) -> str:
        result = self.graph.invoke({"query": arxiv_search_query, "context":context})

        if self.summarize:
            return result.get("final_summary", "No summary generated.")
        else:
            return "\n\nFinished Fetching papers!"
    
    

if __name__ == "__main__":
    agent = ArxivAgent()
    result = agent.run(arxiv_search_query="Experimental Constraints on neutron star radius", 
                       context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?")
    print(result)


