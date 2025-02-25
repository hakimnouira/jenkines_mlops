install:
	pip install -r requirements.txt

prepare:
	python3 main.py --prepare

train:
	python3 main.py --train

evaluate:
	python3 main.py --evaluate

run:
	python3 main.py --run
save:
	python3 main.py --save model.pkl
load:
	python3 main.py --load model.pkl
clean:
	rm -rf __pycache__ *.pkl .pytest_cache

.PHONY: run1
run1:
	uvicorn app:app --reload --host 0.0.0.0 --port 8001

.PHONY: test
test:
	curl -X 'POST' \
	  'http://127.0.0.1:8000/predict' \
	  -H 'Content-Type: application/json' \
	  -d '{"features": [1.5, 2.3, 3.1, 4.7]}' | jq

.PHONY: swagger
swagger:
	xdg-open http://127.0.0.1:8000/docs || start http://127.0.0.1:8000/docs
