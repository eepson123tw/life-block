install:
	cd BE && \
	python -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt && \
	echo "Backend dependencies installed successfully in virtual environment"
	
dev:
	cd BE && \
	. venv/bin/activate && \
	python app.py
