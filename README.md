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

Before using chat, you first need to train sikandar using

```bash
make train
```

the train Makefile target also downloads and prepares the right dataset.

After training you can start an interactive chat session using

```bash
make chat
```

## Running sikandar on Google Colab

You first need to add this repository to your colab notebook

```bash
!git clone SIKANDAR_REPO
```

Then you need to install the requirements:
```bash
%cd sikandar
!pip install -r requirements.txt
```

It is helpful to check if a GPU is available:

```bash
import torch
print(f"gpu available: {torch.cuda.is_available()}")
print(f"gpu name: {torch.cuda.get_device_name(0)}")
```

Finally, we can train using the Google Colab notebook:

```bash
!make train
```

For more intensive training (better chats) you can use the target:

```bash
make train-large
```

We can then download 

# 4. Download model
from google.colab import files
files.download('output/model.pt')
files.download('output/vocab.json')