.PHONY: prepare-data

prepare-data:
	python3 prepare_data.py --output-dir data

.PHONY: train

train: prepare-data
	python3 train.py \
		--train-data data/train.txt \
		--val-data data/val.txt \
		--output-dir output \
		--vocab-size 5000 \
		--d-model 384 \
		--num-heads 6 \
		--num-layers 6 \
		--max-len 256 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--num-epochs 50

.PHONY: chat

chat:
	python3 chat.py \
		--model-path output/model.pt \
		--vocab-path output/vocab.json

.PHONY: generate

generate:
	@echo "Generating text with prompt: $(PROMPT)"
	python3 generate.py \
		--model-path output/model.pt \
		--vocab-path output/vocab.json \
		--prompt "$(PROMPT)" \
		--num-samples 1 \
		--max-tokens 500 \
		--temperature 0.7 \
		--top-k 50 

# Generates from pretrained sikandar model
.PHONY: sikandar

sikandar:
	@python3 generate.py \
		--model-path pretrained/model.pt \
		--vocab-path pretrained/vocab.json \
		--prompt "$(PROMPT)" \
		--num-samples 1 \
		--max-tokens 500 \
		--temperature 0.7 \
		--top-k 50 \
		--log-level ERROR

.PHONY: train-small

train-small: prepare-data
	python3 train.py \
		--train-data data/train.txt \
		--val-data data/val.txt \
		--output-dir output \
		--vocab-size 5000 \
		--d-model 256 \
		--num-heads 8 \
		--num-layers 4 \
		--max-len 128 \
		--batch-size 16 \
		--learning-rate 3e-4 \
		--num-epochs 30

.PHONY: test

test:
	python3 -m unittest discover -s . -p '*_test.py' -v

.PHONY: clean

clean:
	rm -rf output