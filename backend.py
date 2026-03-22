import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

# --- Configuration ---
# 1. Database configuration. Use DB_CONNECTION_STRING to override individual values.
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "").strip()
if not DB_CONNECTION_STRING:
    DB_USER = os.getenv("DB_USER", "philip")
    DB_PASS = os.getenv("DB_PASS", "1234")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "vector_db")
    DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 2. Model + chunking configuration.
MODEL_NAME = os.getenv("MODEL_NAME", "minishlab/potion-multilingual-128M")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "190"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"CHUNK_OVERLAP must be less than CHUNK_SIZE (got {CHUNK_OVERLAP} >= {CHUNK_SIZE})."
    )

# --- Database Setup ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to database. {e}")
        print("Please ensure PostgreSQL is running and the DB_CONNECTION_STRING is correct.")
        return None

def init_db():
    """Initializes the database, creating the vector extension and the documents table."""
    conn = get_db_connection()
    if conn is None:
        return
    
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
        DROP TABLE IF EXISTS documents;
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector({VECTOR_DIMENSION}),
            UNIQUE(file_name, chunk_index)
        );
        """)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# --- Text Processing ---
def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks."""
    chunks = []
    i = 0
    while i < len(text):
        start = i
        end = i + chunk_size
        chunks.append(text[start:end])
        i += (chunk_size - overlap)
    return chunks

# --- Flask App ---
app = Flask(__name__)
CORS(app)  # Allow frontend to call this backend

# Load the embedding model (this will use your VRAM)
print(f"Loading embedding model '{MODEL_NAME}'... (This may take a moment)")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")
VECTOR_DIMENSION = model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {VECTOR_DIMENSION}")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, chunking, embedding, and storing in pgvector."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        content = file.read().decode('utf-8')
        file_name = file.filename
        
        # 1. Chunk the text
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 2. Embed the chunks
        print(f"Embedding {len(chunks)} chunks for '{file_name}'...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        print("Embedding complete.")
        
        # 3. Store in pgvector
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        try:
            with conn.cursor() as cur:
                # Clear old chunks for this file
                cur.execute("DELETE FROM documents WHERE file_name = %s;", (file_name,))

                # Insert new chunks
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    cur.execute(
                        "INSERT INTO documents (file_name, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                        (file_name, i, chunk, embedding)
                    )
            conn.commit()
        finally:
            conn.close()
        
        return jsonify({"message": f"Successfully added {len(chunks)} chunks for {file_name}."}), 200

    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/get-context', methods=['GET'])
def get_context():
    """Gets a user query, embeds it, and finds the top_k relevant chunks."""
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    try:
        # 1. Embed the query
        query_vector = model.encode(query)
        
        # 2. Query pgvector for relevant chunks
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        try:
            with conn.cursor() as cur:
                # Using L2 distance (<->) for similarity search
                cur.execute(
                    """
                    SELECT file_name, chunk_index, content FROM documents
                    ORDER BY embedding <-> %s
                    LIMIT %s;
                    """,
                    (query_vector, TOP_K_CHUNKS)
                )
                results = cur.fetchall()
        finally:
            conn.close()
        
        # 3. Format and return chunks
        retrieved_chunks = [
            {"fileName": r[0], "chunkIndex": r[1], "text": r[2]}
            for r in results
        ]
        
        return jsonify(retrieved_chunks), 200

    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/delete', methods=['POST'])
def delete_file():
    """Deletes all chunks associated with a file."""
    data = request.get_json(silent=True) or {}
    file_name = data.get('fileName')
    if not file_name:
        return jsonify({"error": "No file name provided"}), 400
        
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE file_name = %s;", (file_name,))
            conn.commit()
        finally:
            conn.close()
        
        return jsonify({"message": f"Successfully deleted file {file_name}."}), 200

    except Exception as e:
        print(f"Error during file deletion: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
