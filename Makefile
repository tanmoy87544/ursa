help:
	echo "Help!"

test:
	uv run examples/single_agent_examples/execution_agent/bayesian_optimization.py

test-vowels:
	uv run examples/single_agent_examples/research_agent/ten_vowel_city.py

clean-workspaces:
	rm -rf workspace
	rm -rf workspace_*/
