# How to run ?

## Requirements
Ensure the following is installed.
* [uv](https://docs.astral.sh/uv/)
* [ollama](https://ollama.com)

### Ollama
Download a couple of models by looking up models under [ollama](https://ollama.com/library?sort=popular) and then download them using:

```
ollama pull qwen2.5:7b
```

Verify ollama is running by:

```
ollama serve
```

## Create account with BrightData
Create an account with [BrightData](https://brightdata.com) and get the credentials from SERP API.

## Run the script

```
BD_PORT=<port> BD_USERNAME=<username> BD_PASSWORD=<password> uv run book_writer.py
```

Wait for it to complete, find the final book content under `~/books`.

