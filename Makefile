SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eo pipefail -c

.PHONY: run uninstall install install-pre-commit build deploy

ENV_NAME := hummingbot-api

install:
	@if ! conda env list | sed 's/^\* //' | awk '{print $$1}' | grep -qx $(ENV_NAME); then \
		echo "Creating env $(ENV_NAME) from environment.yml..."; \
		conda env create -f environment.yml; \
	else \
		echo "Environment already exists."; \
	fi
	$(MAKE) install-pre-commit

install-pre-commit:
	@conda run -n $(ENV_NAME) python -c 'import sys; print("Python in env:", sys.executable)'
	@conda run -n $(ENV_NAME) pip install -q pre-commit
	@conda run -n $(ENV_NAME) pre-commit install

run:
	@conda run -n $(ENV_NAME) uvicorn main:app --reload

uninstall:
	@conda env remove -n $(ENV_NAME) -y || true

build:
	@docker build -t hummingbot/hummingbot-api:latest .

deploy:
	@docker compose up -d