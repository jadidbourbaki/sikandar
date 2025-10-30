.PHONY: prepare-data

prepare-data:
	python3 prepare_data.py --output-dir data --max-train 2260 --max-val 119

.PHONY: train

train: prepare-data
	python3 train.py \
		--train-data data/train.txt \
		--val-data data/val.txt \
		--output-dir output \
		--vocab-size 5000 \
		--d-model 128 \
		--num-heads 4 \
		--num-layers 3 \
		--max-len 64 \
		--batch-size 8 \
		--learning-rate 1e-3 \
		--num-epochs 20

.PHONY: chat

chat:
	python3 chat.py \
		--model-path output/model.pt \
		--vocab-path output/vocab.json 

.PHONY: test

test:
	python3 -m unittest discover -s . -p '*_test.py' -v

.PHONY: clean

clean:
	rm -rf output