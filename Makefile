.PHONY: format check-format test clean help all

all: format check-format test clean

format:
	poetry run autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive .
	poetry run isort .
	poetry run black .

check-format:
	poetry run isort --check-only .
	poetry run black --check .

test:
	poetry run pytest tests/

test-all:
	poetry run pytest tests/ --runslow

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +

count:
	find . -path './.venv' -prune -o -name '*.py' -print | xargs wc -l

help:
	@echo "format       - Auto-format Python code with autoflake, isort, and black"
	@echo "check-format - Check if Python code is formatted with isort, and black"
	@echo "test         - Run tests (excluding slow tests)"
	@echo "test-all     - Run all tests (including slow tests)"
	@echo "clean        - Remove all .pyc files and __pycache__ directories"
	@echo "count        - Count the number of lines of code"
	@echo "all          - Run format, check-format, test all and then clean"
