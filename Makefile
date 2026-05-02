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
	python benchmarks/compare_methods.py --datasets swiss_roll --backends tsne --n-seeds 1 --modes direct --n-arrows 1 --skip-significance

bench-full:
	python benchmarks/compare_methods.py --category all --n-seeds 10
