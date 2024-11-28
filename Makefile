.PHONY: requirements
requirements:
	pip install --upgrade pip-tools pip
	pip-compile -v --upgrade \
	requirements/requirements.in \
	requirements/requirements.testing.in \
	-o requirements/requirements.txt

.PHONY: format
format:
	ruff format brightness
