# sikandar llm

## Running sikandar locally

To get started, first create a python environment:

```bash
python3 -m venv .venv
```

then activate it:

```bash
source .venv/bin/activate
```

and finally install the requirements:

```bash
pip install -r requirements.txt
```

Validate the code by running the tests:

```bash
make test
```

Before generating stories, you first need to train sikandar using

```bash
make train
```

the train Makefile target also downloads and prepares the right dataset.

After training you can start generating stories using

```bash
make generate PROMPT="once upon a time"
```

## Running sikandar on Google Colab

You first need to add this repository to your colab notebook. 

Side note: make sure to use the HTTPS url for the repo when working
with Google Colab. Setting up SSH in Colab is kind of tedious.

```bash
!git clone SIKANDAR_REPO
```

Then you need to install the requirements:
```bash
%cd sikandar
!pip install -r colab-requirements.txt --no-deps
```

It is helpful to check if a GPU is available:

```python
import torch
print(f"gpu available: {torch.cuda.is_available()}")
print(f"gpu name: {torch.cuda.get_device_name(0)}")
```

Finally, we can train using the Google Colab notebook:

```bash
!make train
```

We can then download the model

```python
from google.colab import files
files.download('output/model.pt')
files.download('output/vocab.json')
```

or generate the story directly

```bash
!make generate PROMPT="once upon a time"
```