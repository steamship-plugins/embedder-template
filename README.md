# Steamship Embedder Plugin

This project implements a basic Steamship Embedder that you can customize and deploy for your own use.

In Steamship, **Embedders** are responsible for transforming text into vectors that contain meaning in some abstract space.

This sample project transforms raw text into a simple vector based on a bag-of-words inclusion, but other embeders might:

* Call into OpenAI's Embedding endpoints
* Use a BERT-based embedding model
* Average Facebook's FastText word embeddings across a passage

Once an Embedder return vector data to Steamship, it can be used for similarity search and classifier training.

## First Time Setup

We recommend using Python virtual environments for development.
To set one up, run the following command from this directory:

```bash
python3 -m venv .venv
```

Activate your virtual environment by running:

```bash
source .venv/bin/activate
```

Your first time, install the required dependencies with:

```bash
python -m pip install -r requirements.dev.txt
python -m pip install -r requirements.txt
```

## Developing

All the code for this plugin is located in the `src/api.py` file:

* The EmbedderPlugin class
* The `/embed` endpoint

## Testing

Tests are located in the `test/test_api.py` file. You can run them with:

```bash
pytest
```

We have provided sample data in the `test_data/` folder.

## Deploying

Deploy your converter to Steamship by running:

```bash
ship deploy --register-plugin
```

That will deploy your app to Steamship and register it as a plugin for use.

## Using

Once deployed, your Convert Plugin can be referenced by the handle in your `steamship.json` file.

```python
from steamship import Steamship, BlockTypes

MY_PLUGIN_HANDLE = ".. fill this out .."

client = Steamship()
file = client.create_file(file="test_data/facts.txt")
file.convert(plugin=MY_PLUGIN_HANDLE).wait()
file.query(blockType=BlockTypes.Paragraph).wait().data
```

## Sharing

Plesae share what you've built with hello@steamship.com! 

We would love take a look, hear your suggestions, help where we can, and share what you've made with the community.