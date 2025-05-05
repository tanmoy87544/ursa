help:
	echo "Help!"

test:
	uv run examples/example_BO.py

clean-workspaces:
	rm -rf workspace
	rm -rf workspace_*/
