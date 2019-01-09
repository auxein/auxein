.PHONY: help clean

.DEFAULT: help

help:
	@echo "make clean"
	@echo "     Clean up result of build."
	@echo "make build"
	@echo "     Build and package auxein."
	@echo "make release"
	@echo "     Bump up the version and publish Auxein on PyPi."
	@echo "make publish"
	@echo "     Publish Auxein on PyPi."

clean:
	@echo "Cleaning Auxein."
	python setup.py clean --all
	rm -fr *.egg-info
	rm -fr dist

build: clean
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

