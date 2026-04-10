"""Deterministic local embedding stub for E2E testing.

Returns consistent 1536-dimensional unit vectors derived from the SHA-256
hash of the input text. Identical inputs always produce identical vectors,
giving deterministic search ordering in Qdrant.

Compatible with the OpenAI embeddings API format used by the knowledge-api
service (EMBEDDING_BASE_URL / EMBEDDING_API_KEY).

Not a mock — it is a real local embedding provider that produces real vectors
into a real Qdrant instance. Semantic quality is not meaningful, but scope
isolation, auth enforcement, and the ingest→search round-trip are all real.
"""

import hashlib
import json
import math
import os
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer


VECTOR_DIM = 1536


def text_to_unit_vector(text: str) -> list[float]:
    """SHA-256 derived 1536d unit vector — deterministic per input."""
    raw: list[float] = []
    seed = text.encode("utf-8")
    for i in range(VECTOR_DIM):
        digest = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        raw.append(struct.unpack("<f", digest[:4])[0])
    magnitude = math.sqrt(sum(v * v for v in raw))
    return [v / magnitude for v in raw] if magnitude > 0 else raw


class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": text_to_unit_vector(text),
            }
            for i, text in enumerate(inputs)
        ]
        payload = json.dumps({
            "object": "list",
            "data": data,
            "model": body.get("model", "stub-embedding-1536"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # suppress per-request access logs


if __name__ == "__main__":
    port = int(os.getenv("PORT", "9999"))
    server = HTTPServer(("0.0.0.0", port), EmbeddingHandler)
    print(f"embedding-stub: listening on :{port} (dim={VECTOR_DIM})", flush=True)
    server.serve_forever()
