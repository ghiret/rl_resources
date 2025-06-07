![Logo](docs/rl_resources_logo.png)

[![codecov](https://codecov.io/gh/ghiret/rl_resources/graph/badge.svg?token=5LEFWIH4BC)](https://codecov.io/gh/ghiret/rl_resources)
# Deep Reinforcement Learning Resources

This repository is my knowledge repository as I explore and learn Reinforcement learning.
In here you will find code taken from books and adapted during my learning, summaries or notes on papers, and random bits.
The idea of this is to contain my personal notes as I develop a better understanding of the field, primarily for my personal use.
However, this might be helpful for someone else along this journey and that's why I have it in public.

The documentation is published as github pages [here](https://ghiret.github.io/rl_resources/)
## Getting Started

1. Clone this repository
2. Open in VS Code with Dev Containers extension
3. The development container will automatically set up all dependencies using uv
4. Start exploring the tutorials and examples

## Project Structure

```
.
├── .devcontainer/     # Development container configuration
├── docs/             # Documentation source
├── notebooks/        # Jupyter notebooks
├── tests/           # Test files
└── utils/           # Utility functions
```

## Development

### Local Setup

If you prefer to set up the environment locally instead of using the devcontainer:

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create a virtual environment: `uv venv`
3. Activate the virtual environment:
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
4. Install dependencies: `uv pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

### Running Jupyter Notebooks

To run Jupyter notebooks in Cursor:

1. Make sure you have the development dependencies installed:
   ```bash
   uv pip install -e ".[dev]"
   ```

2. Register the virtual environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=rl_resources
   ```

3. Open a notebook in Cursor:
   - Open any `.ipynb` file from the `notebooks/` directory
   - Cursor will automatically detect it as a Jupyter notebook
   - Select the `rl_resources` kernel from the kernel selector in the top-right corner

4. Start coding! You can:
   - Run cells with `Shift + Enter`
   - Add new cells with `B` (below) or `A` (above)
   - Use Markdown cells for documentation
   - Use code cells for Python code

### Code Quality

This project uses pre-commit hooks to ensure code quality:

- **ruff**: Fast Python linter and formatter (replaces Black, isort, and many other tools)
- **pre-commit-hooks**: Various checks for files, YAML, JSON, etc.

To run the checks manually:
```bash
pre-commit run --all-files
```

### Versioning

This project follows [Semantic Versioning](https://semver.org/) principles:

- **MAJOR** version when making incompatible API changes
- **MINOR** version when adding functionality in a backward compatible manner
- **PATCH** version when making backward compatible bug fixes

Releases are automatically managed using [python-semantic-release](https://python-semantic-release.readthedocs.io/).

## Contributing

Feel free to contribute report typos, add feature request or generally asks questions.
I cannot guarantee I will have time to reply, but I will do my best.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## AI Assistance Disclaimer

Please note that a significant portion of the content within this repository (including code, documentation, and configuration) has been generated or assisted by AI tools such as Google Gemini, OpenAI ChatGPT, and Cursor AI. This is done to accelerate development and explore AI capabilities. While the content is reviewed, users should exercise standard diligence and verify its suitability for their needs.

### Logo origins
This was created by ChatGPT after a picture of my dog when she fell asleep on my laptop when I was working on this project.
