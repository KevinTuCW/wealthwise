PY = PYTHONPATH=src .venv/bin/python

install:
	.venv/bin/pip install -e '.[dev]'

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m wealthwise.eval

langfuse-check:
	$(PY) -m wealthwise.langfuse_check

run:
	$(PY) -m uvicorn wealthwise.app:app --reload
