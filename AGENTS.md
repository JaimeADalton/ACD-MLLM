# AGENTS Instructions

This repository contains the ACD-MLLM algorithm. The main entry point is `acd-mllm.py` and the documentation is located in `README.md`.

## Development guidelines

- Use Python 3.8 or newer.
- Keep code style clean. If `flake8` is installed, run `flake8` before committing.
- Always ensure the code compiles:

```bash
python -m py_compile acd-mllm.py
```

There are no test suites, so this compilation check is required before each commit.

## Pull request message

When you open a pull request, summarize your changes and mention if the compilation check succeeded. If any command fails due to environment restrictions, note that in the testing section of the PR message.
