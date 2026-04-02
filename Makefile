PYTHON = python3
PYTHONPATH = PYTHONPATH=.

.PHONY: test test-struct test-valid test-cli lint fmt check verify-1

test:
	$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

test-struct:
	$(PYTHONPATH) $(PYTHON) -m pytest tests/test_structure.py -v

test-valid:
	$(PYTHONPATH) $(PYTHON) -m pytest tests/test_triangle.py tests/test_validation.py -v

test-cli:
	$(PYTHONPATH) $(PYTHON) -m pytest tests/test_cli.py -v

lint: test-struct

fmt:
	$(PYTHON) -m ruff format .

check:
	$(PYTHON) -m ruff check .

verify-1: test
