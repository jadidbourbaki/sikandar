.PHONY: prepare-data

prepare-data:
	python3 prepare_data.py --output-dir data --max-train 0 --max-val 0

.PHONY: train

train: prepare-data
	python3 train.py \
		--train-data data/train.txt \
		--val-data data/val.txt \
		--output-dir output \
		--vocab-size 5000 \
		--d-model 256 \
		--num-heads 8 \
		--num-layers 4 \
		--max-len 128 \
		--batch-size 8 \
		--learning-rate 3e-4 \
		--num-epochs 50

.PHONY: chat

chat:
	python3 chat.py \
		--model-path output/model.pt \
		--vocab-path output/vocab.json

.PHONY: generate

generate:
	@echo "Generating story with prompt: $(PROMPT)"
	python3 generate.py \
		--model-path output/model.pt \
		--vocab-path output/vocab.json \
		--prompt $(PROMPT) \
		--num-samples 1 \
		--max-tokens 200 \
		--temperature 0.7 \
		--top-k 50 

.PHONY: test

test:
	python3 -m unittest discover -s . -p '*_test.py' -v

.PHONY: clean

clean:
	rm -rf output