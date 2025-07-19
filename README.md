# Word Sense Disambiguation

We use spaCy for classical analysis (tokenization, POS tagging, etc.) + 
[entity linking](https://pypi.org/project/spacy-entity-linker/), 
and have our own implementation for [word sense disambiguation](./wsd/word_sense_disambiguation.py).

We expose a [web server](./wsd/server.py) that can be used to disambiguate words in sentences.

### Running locally

```shell
uvicorn --reload wsd.server:app --port 8005 
```

### Running with Docker

```shell
docker build -t wsd .
docker run -p 8000:8000 wsd
```

## Usage

To view an output, visit this [example link](http://localhost:8000/disambiguate?text=Obama%20told%20the%20bus%20driver,%20to%20drive%20to%20D.C.&lang=en&output=html) (adjust port if running locally):
![Example of our system's output](assets/output-example.png)
