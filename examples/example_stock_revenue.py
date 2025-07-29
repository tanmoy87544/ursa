import sys

from langchain_core.messages import HumanMessage
from ursa.agents import ExecutionAgent, PlanningAgent

from langchain_litellm import ChatLiteLLM


# rich console stuff for beautification
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

console = Console()  # global console object


def main(mode: str):
    """Run a simple example of an agent."""
    try:
        # 1. problem statement

        # Nathan:
        # This is the problem description I really want - but I'm definitely having problems
        # consistently getting this to search the internet for companies and then use them
        # it seems to, instead, just fail and then start making up fake companies and wasting
        # tokens playing with fake data.  Replacing this text w/ the one below it:
        # min_year = 2024
        # workspace = "r_and_d_expenditure"
        # problem = (
        #     f"Identify three publicly traded companies with the highest ratio of R&D expenditure "
        #     f"to total revenue in their latest fiscal year (>= {min_year}).  "
        #     f"Search recent SEC 10-K filings or trusted financial APIs for the figures.  "
        #     f"Write Python code to scrape/parse the data, compute the ratios, and present a ranked "
        #     f"DataFrame plus a bar chart of R&D-to-revenue percentages."
        # )
        workspace = "r_and_d_ratio_demo"
        tickers = [
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "META",
            "INTC",
            "JNJ",
        ]
        min_year = 2024
        problem = (
            f"Using SEC EDGAR (or a free financial API), fetch each ticker’s latest fiscal-year "
            f"{min_year}-present Form 10-K: {', '.join(tickers)}.  For every company, obtain total "
            f"revenue and R&D expense, compute the R&D-to-revenue ratio, rank the results highest -> "
            f"lowest, and output a pandas DataFrame.  Save a bar chart to "
            f"rd_ratio.png  **Do not call plt.show(); "
            f"use plt.savefig(...), then plt.close(), so no GUI window appears.**  Code must perform "
            f"a live symbol -> CIK lookup—absolutely no hard-coded CIKs or placeholder keys—and should "
            f"cache the SEC 'company_tickers.json' file locally (e.g., in the workspace) to avoid "
            f"repeated network calls."
        )

        # Nathan needed this - for zscaler.  You might need to remove this if this isn't where you put your
        # zscaler cert stuff.
        problem = problem + (
            "I am on a corporate VPN.  If you need to access the internet, my corporate root CA "
            "certificate is at ~/zscaler_root.pem"
        )

        # print the problem we're solving in a nice little box / panel
        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold cyan]Solving problem:[/] {problem}",
                    justify="center",
                ),
                border_style="cyan",
            )
        )

        # 2. LLM & agents
        model = ChatLiteLLM(
            model="openai/o3-mini"
            # model="openai/o1"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=10000 if mode == "prod" else 4000,
            max_retries=2,
        )

        # Initialize the agents
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

        # 3. top level planning
        # planning agent . . .
        with console.status(
            "[bold green]Planning overarching steps . . .", spinner="point"
        ):
            planning_output = planner.action.invoke(
                {"messages": [HumanMessage(content=problem)]},
                {"recursion_limit": 999999},
            )

        console.print(
            Panel(
                planning_output["messages"][-1].content, title="[yellow]📋 Plan"
            )
        )

        last_step_summary = "Beginning to break down step 1 of the plan."
        detail_planner_prompt = "Flesh out the details of this step and generate substeps to handle the details."

        # 4. do the work...
        # ── OUTER progress bar over main plan steps ─────────────────────────────
        with Progress(
            SpinnerColumn(spinner_name="point"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
            transient=True,
        ) as progress:
            # planning_output is the main planning agent's plans - so they're high level
            # steps that need to be carried out

            # notice top level planning agent steps are then re-planned for more sub-steps by
            # the planning agent again down here.

            main_task = progress.add_task(
                "Main plan steps", total=len(planning_output["plan_steps"])
            )

            # for each of the overarching planning steps . . .
            main_step_number = 1
            for main_step in planning_output["plan_steps"]:
                # ---- detail planning -------------------------------------------------
                step_prompt = (
                    f"You are contributing to the larger solution:\n{problem}\n\n"
                    f"Previous-step summary: {last_step_summary}\n"
                    f"Current step: {main_step}\n\n"
                    f"{detail_planner_prompt}"
                )
                console.print(
                    Panel.fit(
                        Text.from_markup(
                            f"[bold cyan]STEP {main_step_number} - LLM Prompt:[/] {step_prompt}",
                            justify="center",
                        ),
                        border_style="cyan",
                    )
                )

                detail_output = planner.action.invoke(
                    {"messages": [HumanMessage(content=step_prompt)]},
                    {"recursion_limit": 999_999},
                )

                # ---- sub-steps execution --------------------------------------------
                sub_task = progress.add_task(
                    f"Sub-steps for: {str(main_step)[:40]}…",
                    total=len(detail_output["plan_steps"]),
                )

                last_sub_summary = "Start sub-steps."
                sub_step_number = 1
                for sub in detail_output["plan_steps"]:
                    sub_prompt = (
                        f"You are contributing to the larger solution:\n{problem}\n\n"
                        f"Previous-substep summary: {last_sub_summary}\n"
                        f"Current step: {sub}\n\n"
                        "Execute this step and report the results fully—no placeholders."
                    )
                    console.print(
                        Panel.fit(
                            Text.from_markup(
                                f"[bold red]Sub-STEP {sub_step_number} - LLM Prompt:[/] {sub_prompt}",
                                justify="center",
                            ),
                            border_style="red",
                        )
                    )

                    final_results = executor.action.invoke(
                        {
                            "messages": [HumanMessage(content=sub_prompt)],
                            "workspace": workspace,
                        },
                        {"recursion_limit": 999_999},
                    )

                    last_sub_summary = final_results["messages"][-1].content
                    progress.console.log(last_sub_summary)  # live streaming log
                    progress.advance(sub_task)

                    sub_step_number += 1

                progress.remove_task(sub_task)  # collapse bar
                last_step_summary = last_sub_summary
                progress.advance(main_task)
                main_step_number += 1

        # ── 5 · Wrap-up ──────────────────────────────────────────
        answer = last_step_summary
        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold white on green] ✔  Answer:[/] {answer}"
                ),
                border_style="green",
            )
        )
        return answer

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    mode = "dev" if sys.argv[-1] == "dev" else "prod"
    final_output = main(mode=mode)  # dev or prod
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)

    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold white on green] ✔  Answer:[/] {final_output}"
            ),
            border_style="green",
        )
    )

    console.rule("[bold cyan]Run complete")
