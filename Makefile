help:
	echo "Help!"

test:
	uv run examples/example_BO.py

test-vowels:
	uv run examples/example_city_vowels_research.py prod

clean-workspaces:
	rm -rf workspace
	rm -rf workspace_*/
