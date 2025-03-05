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
	uvicorn app:app --reload --host 0.0.0.0 --port 8002

.PHONY: test
test:
	curl -X 'POST' \
	  'http://127.0.0.1:8000/predict' \
	  -H 'Content-Type: application/json' \
	  -d '{"features": [1.5, 2.3, 3.1, 4.7]}' | jq

.PHONY: swagger
swagger:
	xdg-open http://127.0.0.1:8000/docs || start http://127.0.0.1:8000/docs

.PHONY: docker-build
docker-build:
	docker build -t hakim874/hakim_nouira .

.PHONY: docker-push
docker-push:
	docker push hakim874/hakim_nouira

.PHONY: docker-run
docker-run:
	docker run --rm -p 8002:8002 hakim874/hakim_nouira

.PHONY: test

.PHONY: elk-up elk-down check-docker

OS := $(shell uname -s)

# Windows compatibility (Git Bash)
ifeq ($(OS),MINGW64_NT-10.0)
    SHELL := bash
    DOCKER_COMPOSE := winpty docker compose
else
    DOCKER_COMPOSE := docker compose
endif

check-docker:
	@docker info > /dev/null 2>&1 || (echo "❌ Docker is not running. Please start Docker and try again." && exit 1)

elk-up: check-docker
	@echo "🚀 Starting ELK Stack..."
	$(DOCKER_COMPOSE) -f docker-compose-elk.yml up -d
	@echo "✅ ELK Stack is up and running!"

elk-down: check-docker
	@echo "🛑 Stopping ELK Stack..."
	$(DOCKER_COMPOSE) -f docker-compose-elk.yml down
	@echo "✅ ELK Stack has been stopped."
