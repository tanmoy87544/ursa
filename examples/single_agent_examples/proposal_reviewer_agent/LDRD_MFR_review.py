import os  # for envvars from the user for local LLM

import httpx
import litellm
from langchain_litellm import ChatLiteLLM

# rich console stuff for beautification
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from langchain_community.chat_models import ChatLiteLLM
from ursa.agents import ProposalReviewerAgent

# necessary for corporate firewalls / proxies
CA_BUNDLE = os.getenv("CA_BUNDLE")
litellm.client_session = httpx.Client(verify=CA_BUNDLE)

# testing variables
# CFP_pdf = "./FY25_MFR_Problem Statements_FINAL_20240531.pdf"
# review_criteria_json_filename="./proposal_review_criteria_MFR_2025.json",
# trl_levels_json_filename="./TRL_levels.json",
# proposal_dir="../../../../LDRD_Review_MFR_2025/CUI_LDRD_FY25_MFR_PHASE1_INPUT/proposal_PDFs/non-CUI_cover_sheets_and_proposals_70proposals"
# proposal_dir="../../../../LDRD_Review_MFR_2025/CUI_LDRD_FY25_MFR_PHASE1_INPUT/proposal_PDFs/nate_testing"

# production MFR 2026 variables
CFP_pdf = "./FY26_MFR_Problem Statements_June 2 2025.pdf"
review_criteria_json_filename="./proposal_review_criteria_MFR_2026.json"
trl_levels_json_filename="./TRL_levels.json"
proposal_dir="../../../../LDRD_Review_MFR_2026/proposals/"
# proposal_dir="../../../../LDRD_Review_MFR_2026/nate_testing/"

which_endpoint = 'LANL_AI_portal'
# which_endpoint = 'SN'



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
    if which_endpoint == 'SN':
        # this looks really dumb, I realize that, with openai/sambanova/<MODEL> but:
        # When LiteLLM sees a model string of the form provider/model-name, it treats
        # the bit before the first “/” as the provider.
        # LiteLLM’s own docs confirm the prefix is required when you’re talking
        # to SambaNova from a plain OpenAI-style client
        # https://docs.litellm.ai/docs/providers/sambanova
        # - but those docs assume you use the SambaNova provider directly, not a self-hosted
        # OpenAI-compatible endpoint that does its own ACL.
        model="openai/sambanova/Meta-Llama-3.3-70B-Instruct"
        api_base=os.getenv("SN_LLM_URL")
        openai_api_key=os.getenv("SN_LLM_API_KEY")
    elif which_endpoint == 'LANL_AI_portal':
        model="openai/anthropic.claude-3-7-sonnet-20250219-v1:0"
        api_base=os.getenv("LANL_LLM_DEV_BASE")
        openai_api_key=os.getenv("LANL_LLM_DEV_API_KEY")
    else:
        print(f"Invalid endpoint: {which_endpoint}")
        exit()


    # we need to make really sure we are using local LLMs for this so as not to route
    # private info offsite
    model = ChatLiteLLM(
        model=model,
        api_base=api_base,
        openai_api_key=openai_api_key,
        max_tokens=50_000,
        max_retries=2,
        cache=False,
        temperature=0.3, # I think we want SOME temperature
        top_p=0.9, # and a little top_p
    )
    # let's print out the model to give ourselves some confidence

    print_model_info(model)

    agent = ProposalReviewerAgent(llm=model)
    result = agent.run(
        proposal_call_pdf_filename=CFP_pdf,
        review_criteria_json_filename=review_criteria_json_filename,
        trl_levels_json_filename=trl_levels_json_filename,
        submitted_proposals_dir=proposal_dir
    )
    print(result)


if __name__ == "__main__":
    main()
