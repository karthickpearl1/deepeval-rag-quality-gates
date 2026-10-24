# Shortcuts. Each target is a single command you can also run by hand.
export DEEPEVAL_TELEMETRY_OPT_OUT = 1

.PHONY: install lint unit gate gate-regressed gate-deepeval results

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

unit:  ## fast tests: chatbot, PII regex, offline judge, tracing
	pytest --ignore=tests/test_quality_gate.py

gate:  ## the DeepEval quality gate on the good bot (should pass)
	pytest tests/test_quality_gate.py

gate-regressed:  ## the same gate on the regressed bot (should fail)
	BOT_VARIANT=regressed pytest tests/test_quality_gate.py --tb=no

gate-deepeval:  ## the gate through DeepEval's own runner, with its results table
	deepeval test run tests/test_quality_gate.py

results:  ## short summary of the last runs
	python -m evals.show_results good
	python -m evals.show_results regressed
