.PHONY: install-dev test clean bench

install-dev:
	pip install -e ".[dev,pacmap]"
	pip install -r requirements-test.txt

test:
	pytest test/

clean:
	rm -rf build/ dist/ *.egg-info source/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

bench:
	python benchmarks/run_benchmark.py
