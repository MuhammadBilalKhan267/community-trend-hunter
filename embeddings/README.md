# Cached text embeddings

This bundle preserves **27,925 cached vectors** as three lossless compressed NumPy shards, totaling **107,349,223 bytes** (about 102.4 MiB). Each vector has 1,024 float32 dimensions. [manifest.json](manifest.json) records file hashes and row counts.

After installing dependencies, restore the bundle into the runtime cache:

```text
python scripts/embedding_cache.py restore --bundle embeddings --cache .cache/models.sqlite3
```

Restore verifies hashes, dimensions and normalized vectors. It preserves exact float32 bytes and leaves identical entries in place. Encoder profiles and vectors are restored in one transaction; conflicts or invalid data roll back the restore. The bundle excludes generated responses, Jev decisions, credentials and source text.

Each key hashes the encoder name/digest, text and inference options. The bundle includes older entries and the **26,575 distinct August inputs**. A different encoder digest or input produces a cache miss. Earlier cutoff runs still exclude future records from grouping and context.

The evaluated encoder was `bge-m3:latest`, digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`. The manifest's optional `embedding_profiles` metadata records its exact name, digest, semantic options and vector width. Adding this metadata does not change the vector keys or shard bytes.

With `--chat-provider openai`, runs use the recorded profile and cached vectors without contacting Ollama. This also applies to `--prepare-only`, which needs no API keys. The cache-only path pins the recorded encoder revision; it does not query current Ollama tags. Missing profiles, cache misses or corrupt vectors produce `embedding_cache_error` and exit code 2 before any Jev or OpenAI request. There is no inference fallback.

Older bundles remain valid for restore, but cache-only runs require a profile. Restore the updated bundle or run `scripts/ingest_embeddings.py` once with Ollama and BGE available. Live ingestion records the encoder profile after validating results and stores new vectors in the ignored `.cache/` directory. Local-provider runs still use Ollama for installed-model identity and cache misses.

A profile records one current revision per model name and semantic-option set. Live ingestion can refresh this binding after validating results. Restoring a conflicting profile fails; restore that bundle into a fresh cache path instead.

Verification restored all 27,925 entries into an empty cache and compared them byte for byte. The restored cache covered every encoder input at all three evaluation cutoffs. No inference or network requests were made.

To export an updated cache, choose a new destination:

```text
python scripts/embedding_cache.py export --cache .cache/models.sqlite3 --out embeddings-next
```

Export never replaces an existing bundle. It preserves all valid embedding entries and registered profiles, and excludes other cache tables.
