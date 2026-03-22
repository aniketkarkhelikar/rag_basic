# RAG Basic

Simple Retrieval-Augmented Generation (RAG) demo with a Flask backend, PostgreSQL + pgvector for embeddings, and a lightweight HTML/JS chat UI.

## Features
- Upload text files, chunk them, embed, and store in PostgreSQL
- Retrieve the most relevant chunks for a query
- Delete stored file chunks

## Prerequisites
- Python 3.9+
- PostgreSQL with the `pgvector` extension installed
- `psql` CLI available in your PATH

## Quick Start

### 1) Configure environment variables
Copy the example file and adjust values as needed:

```bash
cp .env.example .env
```

Load the variables into your shell (one-time per session):

```bash
set -a
source .env
set +a
```

> You can also export values manually. `DB_CONNECTION_STRING` overrides the individual DB_* values.

### 2) Create the database
Run the setup script (uses sudo to access the `postgres` user):

```bash
bash setupdb.sh
```

You can override the defaults:

```bash
DB_USER=myuser DB_PASS=mypassword DB_NAME=mydb bash setupdb.sh
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Start the backend
```bash
python backend.py
```

The server starts on `http://localhost:5000`.

### 5) Open the frontend
Open `chat.html` in your browser. Add your Gemini API key at the top of the file:

```js
const GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_GOES_HERE";
```

## Configuration
The backend reads configuration from environment variables (see `.env.example`):
- `DB_CONNECTION_STRING` (optional)
- `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`, `DB_NAME`
- `MODEL_NAME`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `TOP_K_CHUNKS`
- `FLASK_DEBUG` (set to `true` to enable debug mode)

## API Endpoints
- `POST /upload` (multipart form-data with `file`)
- `GET /get-context?query=...`
- `POST /delete` (JSON body: `{ "fileName": "example.txt" }`)

## Notes
- The backend recreates the `documents` table on startup, so data is reset each time the server restarts.
- Model downloads can take time on the first run.

## Troubleshooting
- **Database connection failed:** verify PostgreSQL is running and env vars match the created user/db.
- **pgvector extension missing:** install it for your PostgreSQL version, then re-run `setupdb.sh`.
- **Large model download:** the SentenceTransformer model downloads on first run; wait for completion.
