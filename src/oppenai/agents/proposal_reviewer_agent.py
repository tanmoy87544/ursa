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

from typing import Any, Dict, List
from pydantic import BaseModel, Field, create_model
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser,
)

from openai import OpenAI

from .base import BaseAgent
import pathlib, hashlib, os
from pathlib import Path

import time
from datetime import datetime
import pandas as pd
from pydantic import confloat

import json # for prepping for LLM


######################################
# BEGIN: RICH CONSOLE STUFF
######################################
from rich.console import Console
from rich.markdown import Markdown
from rich.theme   import Theme
from rich.rule    import Rule
from rich.panel   import Panel
from rich.syntax import Syntax
from rich.text    import Text
from rich.table import Table 
from rich import box                         # extra box styles (rounded, double…)
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

theme = Theme({
    "markdown.h1":      "bold cyan",
    "markdown.h2":      "bold magenta",
    "markdown.list":    "cyan",
    "markdown.code":    "yellow",
})

console = Console(theme=theme)
######################################
# END: RICH CONSOLE STUFF
######################################


# this is a class we use to force the LLM to conform to JSON structured response
class CFPConformance(BaseModel):
    in_scope: bool = Field(
        description="True if the proposal is in scope for this CFP; else False"
    )
    reasoning: str = Field(
        description="One-paragraph rationale for the in_scope decision"
    )
    topic_areas: list[str] = Field(
        description="List of topic areas that match the CFP (empty list if none)"
    )

# this is a class we use to force the LLM to conform to JSON structured response
class TRLEvaluation(BaseModel):
    trl_level: int = Field(
        description="Integer value representing the TRL level determination using the scale provided"
    )
    reasoning: str = Field(
        description="One-paragraph rationale for the trl_level decision"
    )


# === Data Schemas ===
class ProposalManifest(TypedDict, total=False):
    path: str                # absolute or relative path to the PDF
    filename: str
    raw_text: str
    in_scope: bool              # from the LLM, CFP conformance test
    in_scope_reasoning: str     # from the LLM, CFP conformance test
    topic_areas: List[str]      # from the LLM, CFP conformance test
    trl_level: int
    trl_level_reasoning: str
    # NOTE NOTE NOTE
    # the above is not ALL that's in this 'struct', there will be dynamically
    # added values based on this particular CFP structure - for instance, it
    # might look like this - these are for a particular CFP we are testing with
    # but a different CFP will/might have different scoring areas (e.g. might
    # not be 'technical_vitality' but might be called 'scientific_approach'
    # or something)
    # "technical_vitality_score": 4,
    # "technical_vitality_reasoning": "...",
    # "mission_agility_score": 3,
    # "mission_agility_reasoning": "...",
    # "workforce_development_score": 5,
    # "workforce_development_reasoning": "...",
    # "research_approach_score": 4,
    # "research_approach_reasoning": "..."

class ReviewState(TypedDict, total=False):
    ######################################################
    #  BEGIN: state information related to the CFP
    ######################################################
    proposal_call_pdf_filename: str          # the PDF of the proposal call
    proposal_call_raw_text: str     # the raw text yanked out of the proposal call PDF
    proposal_call_summary: str      # an LLM summary of the proposal call
    ######################################################
    #  END: state information related to the CFP
    ######################################################

    ######################################################
    #  BEGIN: state information related to the TRL levels
    ######################################################
    trl_levels_json_filename: str
    ######################################################
    #  END: state information related to the TRL levels
    ######################################################

    ######################################################
    #  BEGIN: state information related to the submitted proposals
    ######################################################
    submitted_proposals_dir: str    # directory where all the submitted proposal PDFs reside
    proposal_manifests: List[ProposalManifest] # list/array of proposal objects, see above for that structure
    ######################################################
    #  END: state information related to the submitted proposals
    ######################################################

    ######################################################
    #  BEGIN: state information related to the review criteria
    ######################################################
    review_criteria_json_filename: str
    ######################################################
    #  END: state information related to the review criteria
    ######################################################

    rubric: dict                                 # raw rubric JSON (if you keep it)
    proposal_review_parser: PydanticOutputParser # or: Any


# helper functions
def to_pretty_json(obj) -> str:
    # Pydantic v2 first  ➜  use the new API
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=2)

    # Pydantic v1  ➜  json(indent=…)
    if hasattr(obj, "json"):
        try:
            return obj.json(indent=2)
        except TypeError:           # just in case a v2 object slipped through
            return obj.json()

    # Plain dict / list / etc.
    return json.dumps(obj, indent=2)


# === Main Agent ===
class ProposalReviewerAgent(BaseAgent):
    def __init__(self, llm="openai/o3-mini", process_images = True, max_results: int = 3, *args, **kwargs):
        super().__init__(llm, **kwargs)
        self.max_results = max_results
        self.process_images = process_images
        self.graph = self._build_graph()

    def _read_proposal_call_node(self, state: ReviewState) -> ReviewState:
        proposal_call_pdf_filename = state["proposal_call_pdf_filename"]
        print(f"This is proposal call PDF: {proposal_call_pdf_filename}")

        # first, read in the proposal call PDF
        loader = PyPDFLoader(proposal_call_pdf_filename)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])

        return {**state, "proposal_call_raw_text": full_text}
    
    def _summarize_proposal_call_node(self, state: ReviewState) -> ReviewState:
        proposal_call_raw_text = state["proposal_call_raw_text"]
        # print(f"This is proposal call raw text: {proposal_call_raw_text}")

        cfp_prompt = ChatPromptTemplate.from_messages([
            # ----- System role -----
            (
                "system",
                (
                    # Nathan / June, 2025
                    # for the LDRD MFR 2025 call, there are some specific things in the CFP 
                    # and some specific things missing from what I'd call a normal CFP, for instance
                    # there's nothing about who are acceptable people to submit, funding level, etc
                    # so this prompt is pretty specific to that task.
                    "<!-- {unique_query_id} -->"
                    "You are a senior grants analyst with 15 years of experience "
                    "reviewing government and industry Calls-for-Proposals (CFPs).\n"
                    "Your job: distill each CFP-call into a concise, structured briefing "
                    "for principal investigators.\n\n"
                    "You can ignore all names of people and email addresses."
                    "Required output format (in Markdown, in this exact order):\n"
                    "1. Purpose & Background – 2-3 sentences.\n"
                    "2. In-Scope Topics – bullets.\n"
                    "3. For each In-Scope topic, a few paragraphs summarizing the topic itself, "
                    "   a few paragraphs summarizing the background of the topic, "
                    "   and finally a list of R&D priorities."
                    "Do not be TOO concise, we need nuance that was called out in the R&D priorities."
                    "No out of scope topics are specifically called out, but are inferred by "
                    "a list of *IN* scope topics."
                    "Style rules:\n"
                    "* Use crisp bullets, no long paragraphs except in section 1.\n"
                    "* Quote exact wording only when precision matters.\n"
                    "* If the call omits a section, write “*Not specified in the call*”.\n"
                    "* Do not invent or infer beyond the provided text.\n"
                    # this is something that sometimes is done to keep the output clean and
                    # instruct the model not to babble (like I am here) about things like
                    # "ok now i'm reading this, thinking about what the CFP means here . . ."
                    # it sortof kindof works . . . who knows with this stuff?
                    "* Think internally; reveal **only** the final briefing."
                ),
            ),

            # ----- Human role -----
            (
                "human",
                (
                    "Summarize the following CFP-call per the briefing template.\n\n"
                    "<<<CFP-CALL-TEXT-START>>>\n"
                    "{proposal_call_raw_text}\n"
                    "<<<CFP-CALL-TEXT-END>>>"
                ),
            ),
        ])

        # StrOutputParser grabs .content :contentReference[oaicite:1]{index=1}
        chain = cfp_prompt | self.llm | StrOutputParser()   
        def summarize_call(call_text: str) -> str:
            """Run the chain with a Rich spinner."""
            with Progress(
                SpinnerColumn(spinner_name="point", style="cyan"),                 # animated dots
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),                         # shows 0:00 … 0:12
                transient=True,                              # clear the bar when done
                console=console,
            ) as progress:
                task = progress.add_task(f"Summarizing CFP . . .", total=None)
                summary = chain.invoke({
                    "unique_query_id": str(time.time()),
                    "proposal_call_raw_text": call_text
                    })
                progress.update(task, description="[green]✓ CFP summarized")
            return summary


        model_name = getattr(self.llm, "model",        # ChatLiteLLM, ChatOpenAI
                            getattr(self.llm, "model_name", "unknown-model"))

        api_base   = getattr(self.llm, "api_base",     # ChatLiteLLM param
                            getattr(self.llm, "base_url", "unknown-url"))  # OpenAI python-client

        console.print(f'Summarizing using LLM model: [bold cyan]{model_name}')
        console.print(f'            at API endpoint: [bold cyan]{api_base}')
        summary = summarize_call(proposal_call_raw_text)
        console.print("[green]✓ CFP summarized")

        # print("LLM CFP Summary:")
        # print(summary)

        md_renderable = Markdown(summary, hyperlinks=True)
        panel = Panel(
            md_renderable,
            title="CFP Summary",
            title_align="left",          # "center" or "right" also possible
            padding=(1, 2),              # (top_bottom, left_right)
            border_style="bright_green", # any Rich color name
            box=box.ROUNDED,             # box.SQUARE (default), box.DOUBLE, box.HEAVY, …
            expand=False,                # True -> full-width, False -> fit to content
        )
        console.print(panel)

        return {**state, "proposal_call_summary": summary}

    def _collect_proposal_pdfs(self, state: ReviewState) -> ReviewState:
        pdf_dir = pathlib.Path(state["submitted_proposals_dir"])

        manifests: List[ProposalManifest] = []
        for pdf_path in sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            full_text = "\n".join(p.page_content for p in pages)

            manifests.append(
                {
                    "path": str(pdf_path),
                    "filename": pdf_path.name,
                    "raw_text": full_text,
                }
            )
        
        # Build a simple Markdown bullet list of filenames
        filenames_md = "\n".join(f"- `{m['filename']}`" for m in manifests)
        md_renderable = Markdown(filenames_md, hyperlinks=False)

        panel = Panel(
            md_renderable,
            title="Submitted Proposal PDFs",
            title_align="left",
            padding=(1, 2),
            border_style="bright_blue",
            box=box.ROUNDED,
            expand=False,
        )
        console.print(panel)

        return {**state, "proposal_manifests": manifests}
    
    # this is where we evaluate just ONE proposal
    def _evaluate_proposal_for_CFP_conformance(self, manifest, proposal_call_summary):
        # print(manifest['filename'])
        # print(manifest['raw_text'])

        proposal_raw_text = manifest['raw_text']

        # we're going to make the LLM respond in a way we can parse
        parser = PydanticOutputParser(pydantic_object=CFPConformance)
        # this will help FORCE the output into the json format we told it to.
        parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)
        format_instructions = parser.get_format_instructions()

        # here we ask the LLM to review this proposal specifically for conformance to the CFP
        # guidelines
        cfp_prompt = ChatPromptTemplate.from_messages([
            # ----- System role -----
            (
                "system",
                (
                    "<!-- {unique_query_id} -->"
                    "You are a senior grants analyst with 15 years of experience reviewing CFPs.\n"
                    "Read the CFP summary and proposal text, then decide if the proposal is in scope.\n"
                    "Return **only** a JSON object that matches the provided schema; no markdown or extra text."
                    "Style rules:\n"
                    "* Quote exact wording only when precision matters.\n"
                    "* Do not invent or infer beyond the provided text.\n"
                    # this is something that sometimes is done to keep the output clean and
                    # instruct the model not to babble (like I am here) about things like
                    # "ok now i'm reading this, thinking about what the CFP means here . . ."
                    # it sortof kindof works . . . who knows with this stuff?
                    "* Think internally; reveal **only** the final briefing."
                ),
            ),

            # ----- Human role -----
            (
                "human",
                (
                    "Evaluate this proposal for in/out of scope per the instructions below.\n\n"
                    "<<<CFP-CALL-TEXT-START>>>\n{proposal_call_summary}\n<<<CFP-CALL-TEXT-END>>>\n"
                    "<<<PROPOSAL-TEXT-START>>>\n{proposal_raw_text}\n<<<PROPOSAL-TEXT-END>>>\n\n"
                    "{format_instructions}\n"   
                ),
            ),
        ])

        # NOTE HERE - we're using the pydantic parser we detailed above!
        chain = cfp_prompt | self.llm | parser  
        def eval_for_scope(format_instructions : str, proposal_call_summary : str, proposal_raw_text : str) -> str:
            """Run the chain with a Rich spinner."""
            with Progress(
                SpinnerColumn(spinner_name="point", style="cyan"),                 # animated dots
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),                         # shows 0:00 … 0:12
                transient=True,                              # clear the bar when done
                console=console,
            ) as progress:
                task = progress.add_task(f"Evaluating proposal {manifest['filename']} for in/out of scope . . .", total=None)
                summary = chain.invoke({
                    "unique_query_id": str(time.time()),
                    "format_instructions": format_instructions,
                    "proposal_call_summary": proposal_call_summary, 
                    "proposal_raw_text": proposal_raw_text})
                progress.update(task, description="[green]✓ proposal {manifest['filename']} evaluated for in/out of scope")
            return summary


        model_name = getattr(self.llm, "model",        # ChatLiteLLM, ChatOpenAI
                            getattr(self.llm, "model_name", "unknown-model"))

        api_base   = getattr(self.llm, "api_base",     # ChatLiteLLM param
                            getattr(self.llm, "base_url", "unknown-url"))  # OpenAI python-client

        console.print(f'Evaluating proposal vs. CFP scope using LLM model: [bold cyan]{model_name}')
        console.print(f'                                  at API endpoint: [bold cyan]{api_base}')
        eval_response = eval_for_scope(format_instructions, proposal_call_summary, proposal_raw_text)
        console.print(f"[green]✓ proposal {manifest['filename']} evaluated for in/out of scope.")

        json_str = to_pretty_json(eval_response)

        renderable = Syntax(json_str, "json", word_wrap=True)

        print(f"Is this in scope? : {eval_response.in_scope}")

        panel = Panel(
            renderable,
            title="Proposal In/Out-of-Scope Evaluation",
            border_style="bright_green",
            box=box.ROUNDED,
        )
        console.print(panel)

        return eval_response


    # this is where we loop through all the proposals . . .
    def _evaluate_proposals_for_CFP_conformance(self, state: ReviewState) -> ReviewState:
        proposal_call_summary = state["proposal_call_summary"]
        manifests             = state["proposal_manifests"]
        total                 = len(manifests)

        for idx, manifest in enumerate(manifests, start=1):
            console.print(f"[cyan]Processing proposal {idx}/{total} for in/out of scope determination . . .")
            result = self._evaluate_proposal_for_CFP_conformance(
                manifest, proposal_call_summary
            )

            manifest["in_scope"]           = result.in_scope
            manifest["in_scope_reasoning"] = result.reasoning
            manifest["topic_areas"]        = result.topic_areas

        return {**state}


    def _evaluate_proposal_for_TRL_determination(self, manifest, trl_levels_in_json):
        proposal_raw_text = manifest['raw_text']

        # we're going to make the LLM respond in a way we can parse
        parser = PydanticOutputParser(pydantic_object=TRLEvaluation)
        # this will help FORCE the output into the json format we told it to.
        parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm)
        format_instructions = parser.get_format_instructions()

        # here we ask the LLM to review this proposal for TRL determination
        cfp_prompt = ChatPromptTemplate.from_messages([
            # ----- System role -----
            (
                "system",
                (
                    "<!-- {unique_query_id} -->"
                    "You are a senior grants analyst with 15 years of experience reviewing CFPs.\n"
                    "Read the CFP summary and proposal text, then decide the TRL level of the work.\n"
                    "Return **only** a JSON object that matches the provided schema; no markdown or extra text."
                    "Style rules:\n"
                    "* Quote exact wording only when precision matters.\n"
                    "* Do not invent or infer beyond the provided text.\n"
                    # this is something that sometimes is done to keep the output clean and
                    # instruct the model not to babble (like I am here) about things like
                    # "ok now i'm reading this, thinking about what the CFP means here . . ."
                    # it sortof kindof works . . . who knows with this stuff?
                    "* Think internally; reveal **only** the final briefing."
                ),
            ),

            # ----- Human role -----
            (
                "human",
                (
                    "Evaluate this proposal for Technology-Readiness Level (TRL) determination per the instructions below.\n\n"
                    "<<<TRL-LEVELS-JSON>>>\n{trl_levels}\n<<<END TRL-LEVELS-JSON>>>\n"
                    "<<<PROPOSAL>>>\n{proposal_raw_text}\n<<<END PROPOSAL>>>\n\n"
                    "{format_instructions}\n"   
                ),
            ),
        ])

        # NOTE HERE - we're using the pydantic parser we detailed above!
        chain = cfp_prompt | self.llm | parser  
        def eval_for_trl(format_instructions : str, trl_levels_json_str : str, proposal_raw_text : str) -> str:
            """Run the chain with a Rich spinner."""
            with Progress(
                SpinnerColumn(spinner_name="point", style="cyan"),                 # animated dots
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),                         # shows 0:00 … 0:12
                transient=True,                              # clear the bar when done
                console=console,
            ) as progress:
                task = progress.add_task(f"Evaluating proposal {manifest['filename']} for TRL determination . . .", total=None)
                summary = chain.invoke({
                    "unique_query_id": str(time.time()),
                    "format_instructions": format_instructions,
                    "trl_levels": trl_levels_json_str, 
                    "proposal_raw_text": proposal_raw_text})
                progress.update(task, description="[green]✓ proposal {manifest['filename']} evaluated for TRL determination")
            return summary


        model_name = getattr(self.llm, "model",        # ChatLiteLLM, ChatOpenAI
                            getattr(self.llm, "model_name", "unknown-model"))

        api_base   = getattr(self.llm, "api_base",     # ChatLiteLLM param
                            getattr(self.llm, "base_url", "unknown-url"))  # OpenAI python-client

        console.print(f'Evaluating proposal vs. CFP scope using LLM model: [bold cyan]{model_name}')
        console.print(f'                                  at API endpoint: [bold cyan]{api_base}')
        trl_levels_json_str = json.dumps(trl_levels_in_json, indent=2)
        eval_response = eval_for_trl(format_instructions, trl_levels_json_str, proposal_raw_text)
        console.print(f"[green]✓ proposal {manifest['filename']} evaluated for TRL determination.")

        def to_pretty_json(pydantic_obj) -> str:
            # v1 path
            try:
                return pydantic_obj.json(indent=2)
            except TypeError:
                # v2 path
                return pydantic_obj.model_dump_json(indent=2)
        json_str = to_pretty_json(eval_response)

        renderable = Syntax(json_str, "json", word_wrap=True)

        print(f"TRL Determination: {eval_response.trl_level}")

        panel = Panel(
            renderable,
            title="TRL Level Evaluation",
            border_style="bright_green",
            box=box.ROUNDED,
        )
        console.print(panel)

        return eval_response
    
    def _evaluate_proposals_for_TRL_determination(self, state: ReviewState) -> ReviewState:
        trl_levels_json_filename = state["trl_levels_json_filename"]
        manifests             = state["proposal_manifests"]
        total                 = len(manifests)

        def _read_trl_scale(path_str: str) -> dict[int, str]:
            with open(path_str, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}        
        trl_levels_json = _read_trl_scale(trl_levels_json_filename)

        for idx, manifest in enumerate(manifests, start=1):
            console.print(f"[cyan]Processing proposal {idx}/{total} for TRL determination . . .")
            result = self._evaluate_proposal_for_TRL_determination(
                manifest, trl_levels_json
            )

            manifest["trl_level"]           = result.trl_level
            manifest["trl_level_reasoning"] = result.reasoning

        return {**state}
    

    def _read_review_criteria_node(self, state: ReviewState) -> ReviewState:
        review_criteria_json_filename = state["review_criteria_json_filename"]
        print(f"This is review criteria JSON: {review_criteria_json_filename}")

        def _read_review_criteria_json(path_str: str) -> Dict[str, List[dict]]:
            """Return the parsed criteria JSON exactly as stored."""
            path = Path(path_str).expanduser()
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)          # {'categories': [ {...}, {...} ]}
            
        review_criteria_json = _read_review_criteria_json(review_criteria_json_filename)      

        # print("\n=== JSON version ===\n")
        # print(review_criteria_json)

        json_str = to_pretty_json(review_criteria_json)

        renderable = Syntax(json_str, "json", word_wrap=True)

        panel = Panel(
            renderable,
            title="Review Criteria",
            border_style="bright_green",
            box=box.ROUNDED,
        )
        console.print(panel)

        # --- build the dynamic ProposalReview model --------------------------
        rubric = review_criteria_json
        fields: Dict[str, tuple[Any, Field]] = {
            "summary": (str, Field(..., description="One-paragraph proposal summary"))
        }

        for cat in rubric["categories"]:
            safe = cat["name"].lower().replace(" ", "_")
            fields[f"{safe}_score"] = (
                confloat(multiple_of=0.1),  # 0.0, 0.1, …, 9.9
                Field(..., description=f"{cat['name']} numeric score"),
            )
            fields[f"{safe}_reasoning"] = (
                str,
                Field(..., description=f"{cat['name']} score rationale"),
            )

        ProposalReview = create_model("ProposalReview", **fields)  # <- dynamic class
        parser         = PydanticOutputParser(pydantic_object=ProposalReview)

        schema = ProposalReview.model_json_schema()      # v2 (use .schema() in v1)
        json_str = json.dumps(schema, indent=2)

        console.print(
            Panel(
                Syntax(json_str, "json", word_wrap=True),
                title="Dynamic ProposalReview schema",
                border_style="cyan"
            )
        )

        # --- keep both the raw rubric & the parser in state ------------------
        return {
            **state,
            "rubric": rubric,
            "proposal_review_parser": parser,      # anything downstream can import this
        }
    
    def _evaluate_proposal_for_review(
        self,
        manifest: ProposalManifest,
        rubric: dict,
        proposal_review_parser,          # <- the parser you stored in state
    ):
        """LLM-evaluate one proposal against the rubric and return a ProposalReview model."""

        # -------------------------------------------------------
        # 1.  Pull proposal text
        # -------------------------------------------------------
        proposal_raw_text = manifest["raw_text"]

        # -------------------------------------------------------
        # 2.  Wrap the parser with OutputFixingParser
        #     (so we auto-repair bad JSON)
        # -------------------------------------------------------
        parser_base  = proposal_review_parser          # dynamic Pydantic parser
        parser       = OutputFixingParser.from_llm(parser=parser_base, llm=self.llm)
        fmt_instr    = parser_base.get_format_instructions()

        # -------------------------------------------------------
        # 3.  Build the prompt
        # -------------------------------------------------------
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "<!-- {unique_query_id} --> "
                        "You are a senior proposal reviewer. Use the rubric provided to score "
                        "each category and give two-paragraph rationales. "
                        "Please be critical - harsh when necessary, but not needlessly cruel.  "
                        "Please be *specific*, we are not interested in general vague responses - "
                        "we need reviews that are actionable and critical for the authors.  "
                        "Do not simply review everything at the highest level unless it really "
                        "warrants it.  "
                        "Reply ONLY with a JSON object that matches the schema; "
                        "no markdown or extraneous text."
                    ),
                ),
                (
                    "human",
                    (
                        "<<<RUBRIC>>>\\n{rubric_json}\\n<<<END RUBRIC>>>\\n"
                        "<<<PROPOSAL>>>\\n{proposal_raw_text}\\n<<<END PROPOSAL>>>\\n\\n"
                        "{format_instructions}"
                    ),
                ),
            ]
        )

        # -------------------------------------------------------
        # 4.  Build the runnable chain
        # -------------------------------------------------------
        chain = prompt | self.llm | parser

        # -------------------------------------------------------
        # 5.  Run with a nice spinner
        # -------------------------------------------------------
        rubric_json_str = json.dumps(rubric, indent=2)

        with Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task(
                f"Reviewing {manifest['filename']} …", total=None
            )
            result = chain.invoke(
                {
                    "unique_query_id": str(time.time()),
                    "rubric_json": rubric_json_str,
                    "proposal_raw_text": proposal_raw_text,
                    "format_instructions": fmt_instr,
                }
            )
            prog.update(task, description=f"[green]✓ {manifest['filename']} reviewed")

        # -------------------------------------------------------
        # 6.  Pretty-print the JSON and show a panel (optional)
        # -------------------------------------------------------
        json_str = json.dumps(result.model_dump(), indent=2)   # v2; use .dict() in v1

        console.print(
            Panel(
                Syntax(json_str, "json", word_wrap=True),
                title=f"Full Review – {manifest['filename']}",
                border_style="bright_green",
                box=box.ROUNDED,
            )
        )

        return result        # a ProposalReview pydantic object

        
    def _evaluate_proposals_for_review(self, state: ReviewState) -> ReviewState:
        manifests             = state["proposal_manifests"]
        total                 = len(manifests)
        rubric = state['rubric']
        proposal_review_parser = state['proposal_review_parser']

        for idx, manifest in enumerate(manifests, start=1):
            console.print(f"[cyan]Processing proposal {idx}/{total} for full review . . .")
            result = self._evaluate_proposal_for_review(
                manifest, rubric, proposal_review_parser
            )

            manifest.update(result.model_dump()) # copies the dynamic fields in

        return {**state}


    def _aggregate_node(self, state: ReviewState) -> ReviewState:
        console.print("[bold underline cyan]\n=== Proposals Analysis ===\n")

        rubric = state.get("rubric", {})          # keep the dynamic bit flexible
        categories = rubric.get("categories", [])

        # pre-compute the safe field names that the LLM/model used
        cat_fields = [
            cat["name"].lower().replace(" ", "_")  # e.g. technical_vitality
            for cat in categories
        ]
    
        for m in state["proposal_manifests"]:
            # ----------- build the inner table ----------
            grid = Table.grid(padding=(0, 1))
            grid.expand = False                           # fit to content

            # row: Topic Areas
            topics = ", ".join(m["topic_areas"]) or "—"
            grid.add_row("Topic Areas:", Text(topics, style="yellow"))

            # row: In-scope flag
            flag = Text("YES", style="bold green") if m["in_scope"] else Text("NO", style="bold red")
            grid.add_row("In Scope?:", flag)

            # row: Reasoning (wrapped; muted colour)
            reasoning = Text(
                m["in_scope_reasoning"],
                style="bright_white",
                overflow="fold"          # ← this does the wrapping
            )
            grid.add_row("Scope Reasoning:", reasoning)

            trl_level     = m.get("trl_level", "—")
            trl_row_style = "bold green" if isinstance(trl_level, int) and trl_level >= 3 else "bold red"
            grid.add_row("TRL Level:", Text(str(trl_level), style=trl_row_style))

            trl_reason = Text(m.get("trl_level_reasoning", "—"), style="bright_white", overflow="fold")
            grid.add_row("TRL Reasoning:", trl_reason)


            # ---------- dynamic rubric rows ----------
            for safe, cat in zip(cat_fields, categories):
                score_key      = f"{safe}_score"
                reason_key     = f"{safe}_reasoning"
                score          = m.get(score_key, "—")
                reasoning_text = m.get(reason_key, "—")

                # colour score by threshold
                if isinstance(score, (int, float)):
                    score_style = (
                        "bold green"  if score >= 4 else
                        "bold yellow" if score == 3 else
                        "bold red"
                    )
                else:
                    score_style = "bright_white"

                grid.add_row(f"{cat['name']} Score:",
                            Text(str(score), style=score_style))
                grid.add_row(f"{cat['name']} Reason:",
                            Text(reasoning_text, style="bright_white", overflow="fold"))


            # ----------- wrap table in a panel ----------
            panel = Panel(
                grid,
                title=f"[bold]{m['filename']}",
                border_style="bright_green" if m["in_scope"] else "bright_red",
                box=box.ROUNDED,
                padding=(1, 2),
                expand=False,
            )
            console.print(panel)

        return {**state}
    

    def _export_reviews_to_csv(self, state: ReviewState) -> ReviewState:
        manifests = state["proposal_manifests"]

        # Build DataFrame from ALL keys
        df = pd.DataFrame(manifests)

        # we don't want these fields
        for col in ("raw_text","path"):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Flatten topic_areas list → comma-separated string
        if "topic_areas" in df.columns:
            df["topic_areas"] = df["topic_areas"].apply(
                lambda lst: ", ".join(lst) if isinstance(lst, list) else lst
            )

        # Timestamped filename
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_path = Path(f"proposals_review_{ts}.csv").resolve()
        df.to_csv(out_path, index=False)

        console.print(
            Panel(
                f"[green]✓ CSV exported to {out_path}",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # keep path in state if later nodes need it
        return {**state, "reviews_csv": str(out_path)}
    
        
    def _build_graph(self):
        builder = StateGraph(ReviewState)
        
        # reads the proposal PDF in
        builder.add_node("read_proposal_call", self._read_proposal_call_node)
        
        # uses the LLM to summarize the CFP into topics, R&D targets, etc
        builder.add_node("summarize_proposal_call", self._summarize_proposal_call_node)

        # let's start looking at the proposals themselves that were submitted for review
        builder.add_node("collect_proposal_pdfs", self._collect_proposal_pdfs)
        # this node evaluates a single proposal, seeing if it's responsive to the call
        builder.add_node("evaluate_proposals_for_CFP_conformance", self._evaluate_proposals_for_CFP_conformance)
        builder.add_node("evaluate_proposals_for_TRL_determination", self._evaluate_proposals_for_TRL_determination)

        # reads the proposal review criteria
        builder.add_node("read_review_criteria", self._read_review_criteria_node)
        builder.add_node("evaluate_proposals_for_review", self._evaluate_proposals_for_review)

        builder.add_node("aggregate", self._aggregate_node)
        builder.add_node("export_to_csv", self._export_reviews_to_csv)


        builder.set_entry_point("read_proposal_call")

        # step 1: read the proposal call PDF into raw text
        # builder.set_entry_point("read_proposal_call")
        # step 2: after we have the proposal call raw text, ask the LLM to summarize it
        builder.add_edge("read_proposal_call", "summarize_proposal_call")
        # step 3: let's work on the proposals - first, we need to collect all the proposals
        builder.add_edge("summarize_proposal_call", "collect_proposal_pdfs")
        # step 4: let's get all of our proposals together
        builder.add_edge("collect_proposal_pdfs", "evaluate_proposals_for_CFP_conformance")
        # step 5: next let's evaluate all the proposals for in/out of scope
        builder.add_edge("evaluate_proposals_for_CFP_conformance", "evaluate_proposals_for_TRL_determination")
        # step 6: then determine the TRL
        builder.add_edge("evaluate_proposals_for_TRL_determination", "read_review_criteria")
        builder.add_edge("read_review_criteria", "evaluate_proposals_for_review")
        builder.add_edge("evaluate_proposals_for_review", "aggregate")
        builder.add_edge("aggregate", "export_to_csv")
        builder.set_finish_point("export_to_csv")


        graph = builder.compile()
        return graph

    def run(self, 
            proposal_call_pdf_filename: str,
            review_criteria_json_filename: str, 
            trl_levels_json_filename: str, 
            submitted_proposals_dir: str, 
            recursion_limit=100) -> str:
        # we need to stuff this input information into our state graph so nodes
        # can slurp it up
        result = self.graph.invoke(
            {
                "proposal_call_pdf_filename": proposal_call_pdf_filename,
                "review_criteria_json_filename": review_criteria_json_filename,
                "trl_levels_json_filename" : trl_levels_json_filename,
                "submitted_proposals_dir" : submitted_proposals_dir,
            },
            {"recursion_limit":recursion_limit},
        )
        # result = self.graph.invoke({"query": arxiv_search_query, "context":context}, {"recursion_limit":recursion_limit})
        # return result.get("final_summary", "No summary generated.")



if __name__ == "__main__":
    print("This agent doesn't have any defaults at this time.  Exiting.")

