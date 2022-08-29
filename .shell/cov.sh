#!/bin/bash
find . -name 'coverage.txt' -delete
poetry run pytest --cov-report term --cov ner_api tests/ >>.logs/coverage.txt
