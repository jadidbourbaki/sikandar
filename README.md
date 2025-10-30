# sikandar llm


## Running sikandar locally

To get started, first create a python environment:

```
python3 -m venv .venv
```

then activate it:

```
source .venv/bin/activate
```

and finally install the requirements:

```
pip install -r requirements.txt
```

Validate the code by running the tests:

```
make test
```

Before using chat, you first need to train sikandar using

```
make train
```

the train Makefile target also downloads and prepares the right dataset.

After training you can start an interactive chat session using

```
make chat
```