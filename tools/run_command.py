import subprocess

from langchain_core.tools    import tool

@tool
def run_cmd(query: str, workspace_dir, model, timeout=600):
    """Run command from commandline"""
    safety_check = model.invoke("Assume commands to run python are safe because the files are from a trusted source. Answer only either [YES] or [NO]. Is this command safe to run: " + query)
    if "[NO]" in safety_check.content:
        print("[WARNING]")
        print("[WARNING] That command deemed unsafe and cannot be run: ", query, " --- ",safety_check)
        print("[WARNING]")
        return ["[WARNING] That command deemed unsafe and cannot be run."]
    
    print("RUNNING: ", query)
    process = subprocess.Popen(
        query.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace_dir
    )

    stdout, stderr = process.communicate(timeout=timeout)

    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"
