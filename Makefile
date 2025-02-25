install:
	cd BE && \
	python -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt && \
	echo "Backend dependencies installed successfully in virtual environment"
dev:
	uvicorn src.main:app --host 0.0.0.0  --port 8001 --workers 2

