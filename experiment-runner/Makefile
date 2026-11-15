.PHONY: build up down logs train-tfidf run-rules run-qwen

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

train-tfidf:
	docker compose exec api phishlab train-tfidf-colab --name phishfuzzer-all-42

run-rules:
	docker compose exec api phishlab run --config configs/baselines/rules.yaml

run-qwen:
	docker compose exec api phishlab evaluate-adapter --config configs/llm/qwen3-4b-base-4bit-phishing-origin.yaml

