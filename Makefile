.PHONY: clean lint

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint:
	isort algo_gen/
	autoflake --in-place algo_gen/*.py
	black algo_gen/
	flake8 algo_gen/
	pylint algo_gen/
	mypy algo_gen/
