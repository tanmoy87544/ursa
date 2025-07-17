import sys
import os # for envvars from the user for local LLM

# rich console stuff for beautification
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from langchain_community.chat_models import ChatLiteLLM
from ursa.agents import ProposalReviewerAgent

import litellm, httpx

# necessary for corporate firewalls / proxies
CA_BUNDLE = os.getenv("CA_BUNDLE")
litellm.client_session = httpx.Client(verify=CA_BUNDLE)

proposal_dir="../../../../LDRD_Review_MFR_2025/CUI_LDRD_FY25_MFR_PHASE1_INPUT/proposal_PDFs/non-CUI_cover_sheets_and_proposals_70proposals"
# proposal_dir="../../../../LDRD_Review_MFR_2025/CUI_LDRD_FY25_MFR_PHASE1_INPUT/proposal_PDFs/nate_testing"

console = Console()          # global console object

def print_model_info(model):
    def mask_key(key: str, visible: int = 8) -> str:
        """Return the first `visible` chars of `key`, then an ellipsis."""
        return key[:visible] + ". . ." if key else ""
    
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="bold cyan")
    grid.add_column()

    grid.add_row("Model", model.model)
    grid.add_row("API base", model.api_base)
    grid.add_row("API key", mask_key(model.openai_api_key))
    console.print(
        Panel.fit(
            grid,
            title=Text.from_markup("[bold cyan]LLM Configuration[/]"),
            border_style="cyan",
        )
    )


def main():
    # we need to make really sure we are using local LLMs for this so as not to route
    # private info offsite
    model = ChatLiteLLM(
        # this looks really dumb, I realize that, with openai/sambanova/<MODEL> but:
        # When LiteLLM sees a model string of the form provider/model-name, it treats 
        # the bit before the first “/” as the provider.
        # LiteLLM’s own docs confirm the prefix is required when you’re talking 
        # to SambaNova from a plain OpenAI-style client
        # https://docs.litellm.ai/docs/providers/sambanova
        # - but those docs assume you use the SambaNova provider directly, not a self-hosted 
        # OpenAI-compatible endpoint that does its own ACL.
        model="openai/sambanova/Meta-Llama-3.3-70B-Instruct", # or any model you deem appropriate for this
        api_base=os.getenv("SN_LLM_URL"), # the SambaNova endpoint / local for us
        openai_api_key=os.getenv("SN_LLM_API_KEY"), # your API key
        max_tokens=50_000,
        max_retries=2,
        cache=False,
        # I can't get any of these temperature values to mean a damned thing w/
        # the llama 3.3 70B instruct
        # temperature=0.8,
        # top_k=50, top_p=1.0,
    )
    # let's print out the model to give ourselves some confidence

    print_model_info(model)

    agent = ProposalReviewerAgent(llm=model)
    result = agent.run(
        proposal_call_pdf_filename="./FY25_MFR_Problem Statements_FINAL_20240531.pdf",
        review_criteria_json_filename="./proposal_review_criteria.json",
        trl_levels_json_filename="./TRL_levels.json",
        submitted_proposals_dir=proposal_dir
    )
    print(result)


if __name__ == "__main__":
    main()
