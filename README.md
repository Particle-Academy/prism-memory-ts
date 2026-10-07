# Prism Memory for TypeScript

Persistent context, semantic recall, and the vector-store contract. The
TypeScript port of
[`particle-academy/prism-memory`](https://github.com/Particle-Academy/prism-memory).

Zero runtime dependencies. Node 22+.

```
npm install @particle-academy/prism-memory
```

## Usage

```ts
import {
  InMemoryVectorStore,
  RecallSettings,
  Vector,
  recall,
} from '@particle-academy/prism-memory';

const store = new InMemoryVectorStore();

await store.upsert([
  {
    id: 'note-1',
    collection: 'support',
    content: 'The customer runs Postgres 16.',
    kind: 'observation',
    vector: Vector.of([0.21, -0.04, 0.88]),
    embeddingModel: 'text-embedding-3-small',
    metadata: { ticket: 'T-914' },
    provenance: { source: 'thread-88', author: 'customer', observedAt: 1_760_000_000 },
    createdAt: 1_760_000_000,
  },
]);

const recollection = await recall(
  store,
  { collections: ['support'], vector: Vector.of(queryEmbedding) },
  new RecallSettings(),
);
```

`recall()` asks the store for a candidate budget, then scores what came back by
similarity **and age** — so a store that only knows about distance still returns
results that prefer what was observed recently. `Weighting` is where that
trade-off lives, and it is a value object rather than a pair of tuning
constants, because the same weighting has to be expressible in three languages.

Implement `VectorStore` — `upsert`, `search`, `durability` — for pgvector,
Qdrant, Pinecone, or whatever you already run.

## The contract has one hard rule

**Declare your own `durability()`.** `InMemoryVectorStore` reports `volatile`
and is right to: it is for tests and single-process tools. Only you know whether
your Redis is persistent or a disposable cache, and that declaration is an
assertion about your infrastructure, not a preference.

**Two embedding spaces must never be mixed.** A record carries the
`embeddingModel` that produced its vector. Cosine distance between vectors from
different models is a number, and it is meaningless — which is worse than an
error, because it ranks.

A vector must have at least one component; an empty one throws `MemoryError`
with code `invalid_vector` rather than scoring as maximally distant from
everything.

## Parity

`Vector.fromStorage()` / `toStorage()` use the same base64-of-float64 encoding
as the PHP reference and the Python port, so one store can be written by a PHP
app and read by a TypeScript agent. A length that is not a whole number of
8-byte components is refused rather than read as a shorter vector. The scoring
and the recall settings are pinned by prism-parity's memory corpus, which found
and closed G-22 on the day it was written.
