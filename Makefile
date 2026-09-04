.PHONY: install test lint format notebook render clean

install:
	pip install -e ".[dev,data,notebook]"

test:
	pytest --cov=bsm --cov-report=term-missing

lint:
	ruff check bsm tests

format:
	ruff check --fix bsm tests
	ruff format bsm tests

notebook:
	jupyter lab notebooks/black_scholes_spy.ipynb

# Execute the notebook and export a static HTML report to docs/
render:
	jupyter nbconvert --to notebook --execute --inplace notebooks/black_scholes_spy.ipynb
	jupyter nbconvert --to html notebooks/black_scholes_spy.ipynb \
		--output-dir docs --output black_scholes_spy.html

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov **/__pycache__ *.egg-info
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
