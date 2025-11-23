export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./.venv/bin/python -m pytest --cov=src/axiom tests/
