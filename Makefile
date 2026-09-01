PY = PYTHONPATH=src .venv/bin/python

install:
	.venv/bin/pip install -e '.[dev]'

test:
	$(PY) -m pytest -q

# Postgres RunStore integration tests. They skip without a DSN, so the default
# `make test` stays offline and needs no database.
#   createdb wealthwise && make test-pg
test-pg:
	WEALTHWISE_TEST_PG_DSN=$${WEALTHWISE_TEST_PG_DSN:-postgresql:///wealthwise} \
	  $(PY) -m pytest tests/test_store.py -q

eval:
	$(PY) -m wealthwise.eval

# Factor-weight backtest over the shipped universe. Needs network on the first
# run; the fetched bars are cached under /tmp afterwards.
backtest:
	$(PY) scripts/backtest_factors.py

langfuse-check:
	$(PY) -m wealthwise.langfuse_check

run:
	$(PY) -m uvicorn wealthwise.app:app --reload
