.PHONY: help clean

.DEFAULT: help

help:
	@echo "make setup"
	@echo "     Setting up all the dependencies."
	@echo "make clean"
	@echo "     Clean up result of build."
	@echo "make codestyle"
	@echo "     Run code style check."
	@echo "make typecheck"
	@echo "     Run type check."
	@echo "make test"
	@echo "     Run unit tests."
	@echo "make build"
	@echo "     Build and package auxein."
	@echo "make release"
	@echo "     Bump up the version and publish Auxein on PyPi."
	@echo "make publish"
	@echo "     Publish Auxein on PyPi."

setup:
	@echo "Setting up the environment."
	pip install -e .[tests]

clean:
	@echo "Cleaning Auxein."
	python setup.py clean --all
	rm -fr *.egg-info
	rm -fr dist

test:
	@echo "Running unit tests."
	python -m pytest --cov=auxein tests/

typecheck:
	@echo "Running type check."
	python -m mypy --strict --config-file=mypy.ini auxein/

codestyle:
	@echo "Running code style check."
	python -m flake8

build: clean codestyle typecheck test
	@echo "Build Auxein."
	python setup.py sdist bdist_wheel

publish: build
	@echo "Publishing Auxein to PyPi."
	python -m twine upload dist/*

AUXEIN_VERSION=$(shell python setup.py --version)

release: publish
	@echo "Releasing Auxein $(AUXEIN_VERSION)"
	bumpversion --current-version $(AUXEIN_VERSION) minor setup.py
	git add setup.py
	@echo "Current branch is now at $(shell python setup.py --version)"

