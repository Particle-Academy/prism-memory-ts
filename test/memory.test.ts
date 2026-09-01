import { Buffer } from 'node:buffer';
import { describe, expect, it } from 'vitest';
import {
  BinarySignature,
  InMemoryVectorStore,
  MemoryError,
  RecallSettings,
  Vector,
  Weighting,
  recall,
  type VectorRecord,
} from '../src/index.js';

function record(overrides: Partial<VectorRecord> = {}): VectorRecord {
  return {
    id: 'm-1',
    collection: 'default',
    content: 'something that was said',
    kind: 'observation',
    vector: Vector.of([1, 0, 0]),
    embeddingModel: 'text-embedding-3-small',
    metadata: {},
    provenance: { source: null, author: null, observedAt: 1000 },
    createdAt: 1000,
    ...overrides,
  };
}

describe('Vector storage', () => {
  it('round-trips through the stored form EXACTLY', () => {
    // float32 would lose roughly nine significant digits per component, and the
    // ranking would drift by an amount no "did it come back" test would notice.
    const values = [0.1, -0.2, 1e-17, 12345.6789012345, Math.PI];
    const restored = Vector.fromStorage(Vector.of(values).toStorage());

    expect(restored.values).toEqual(values);
  });

  it('stores LITTLE-ENDIAN doubles, byte for byte what the reference writes', () => {
    // The parity claim: a PHP app and this port read each other's rows. `pack('e*')`
    // is little-endian float64, and that is what this must produce.
    const encoded = Vector.of([1]).toStorage();
    const bytes = Buffer.from(encoded, 'base64');

    expect(bytes.length).toBe(8);
    expect([...bytes]).toEqual([0, 0, 0, 0, 0, 0, 0xf0, 0x3f]);
  });

  it('REFUSES a non-finite component at the write path', () => {
    // A single NaN inside a stored vector makes every score against it NaN, and
    // NaN comparisons are false — so the record silently stops being
    // retrievable rather than failing.
    expect(() => Vector.of([1, Number.NaN, 3])).toThrowError(/NaN/);
    expect(() => Vector.of([1, Number.POSITIVE_INFINITY])).toThrowError(MemoryError);
  });

  it('refuses an empty vector', () => {
    expect(() => Vector.of([])).toThrowError(/at least one component/);
  });

  it('refuses a stored blob that is not a whole number of doubles', () => {
    expect(() => Vector.fromStorage(Buffer.from([1, 2, 3]).toString('base64'))).toThrowError(
      /8-byte doubles/,
    );
  });

  it('parses numeric strings, which is what a JSON column gives back', () => {
    expect(Vector.of(['0.5', '0.25']).values).toEqual([0.5, 0.25]);
  });
});

describe('cosine', () => {
  it('is 1 for the same direction and 0 for orthogonal', () => {
    expect(Vector.of([1, 0]).cosine(Vector.of([2, 0]))).toBeCloseTo(1);
    expect(Vector.of([1, 0]).cosine(Vector.of([0, 1]))).toBeCloseTo(0);
    expect(Vector.of([1, 0]).cosine(Vector.of([-1, 0]))).toBeCloseTo(-1);
  });

  it('stays inside [-1, 1] even where floating point would not', () => {
    // The RANGE is a guarantee even where the last bit is not.
    const similarity = Vector.of([0.1, 0.2, 0.3]).cosine(Vector.of([0.1, 0.2, 0.3]));

    expect(similarity).toBeLessThanOrEqual(1);
    expect(similarity).toBeGreaterThanOrEqual(-1);
  });

  it('names a dimension mismatch rather than comparing what it can', () => {
    expect(() => Vector.of([1, 2]).cosine(Vector.of([1, 2, 3]))).toThrowError(/dimension/);
  });

  it('refuses a zero vector, which has no direction', () => {
    expect(() => Vector.of([0, 0]).cosine(Vector.of([1, 0]))).toThrowError(/no direction/);
  });

  it('survives a PRODUCT that overflows while each squared length is finite', () => {
    // The case the fallback exists for: 1e150 squared is 1e300, so each vector's
    // own sum of squares is finite, and their product is not. Rarer than it
    // sounds and not impossible; Infinity would cost every bit.
    const huge = Vector.of([1e150, 1e150]);

    expect(huge.cosine(huge)).toBeCloseTo(1);
  });

  it('refuses a vector too large to score AT ALL, distinctly from a zero one', () => {
    // Two genuinely different failures. Collapsing them into one message would
    // send whoever reads it looking in the wrong place.
    expect(() => Vector.of([1e200, 1e200]).cosine(Vector.of([1, 1]))).toThrowError(/overflows/);
    expect(() => Vector.of([0, 0]).cosine(Vector.of([1, 1]))).toThrowError(/no direction/);
  });
});

describe('BinarySignature', () => {
  it('is DETERMINISTIC for the same seed', () => {
    // The planes must be identical everywhere the same collection is read, or
    // two processes would compute different signatures for the same vector and
    // neither would be wrong — recall would simply stop finding things.
    const vector = Vector.of([0.3, -0.7, 0.1, 0.9]);

    expect(new BinarySignature(4, 64).of(vector)).toBe(new BinarySignature(4, 64).of(vector));
  });

  it('gives similar vectors a smaller Hamming distance than dissimilar ones', () => {
    // Hamming distance over `bits` estimates θ/π: that is what makes the
    // signature enough to RANK candidates without reading the vectors.
    const signature = new BinarySignature(3, 256);
    const query = Vector.of([1, 0, 0]);

    const near = BinarySignature.distance(signature.of(query), signature.of(Vector.of([0.99, 0.1, 0])));
    const far = BinarySignature.distance(signature.of(query), signature.of(Vector.of([-1, 0, 0])));

    expect(near).toBeLessThan(far);
  });

  it('packs one bit per plane', () => {
    const encoded = new BinarySignature(3, 64).of(Vector.of([1, 1, 1]));

    expect(Buffer.from(encoded, 'base64').length).toBe(8);
  });

  it('refuses a bit count that is not a whole number of bytes', () => {
    expect(() => new BinarySignature(3, 100)).toThrowError(MemoryError);
  });

  it('refuses to compare signatures of different lengths', () => {
    const a = new BinarySignature(3, 64).of(Vector.of([1, 0, 0]));
    const b = new BinarySignature(3, 128).of(Vector.of([1, 0, 0]));

    expect(() => BinarySignature.distance(a, b)).toThrowError(/different lengths/);
  });
});

describe('Weighting', () => {
  it('defaults to relevance alone', () => {
    // What someone expects from something called semantic recall.
    const weighting = new Weighting();

    expect(weighting.usesRecency()).toBe(false);
    expect(weighting.score(0.8, 999_999)).toBeCloseTo(0.8);
  });

  it('NORMALISES, so a score stays in range whatever the mix', () => {
    // Without this, raising the recency weight would raise every score and
    // quietly disable the caller's minScore threshold.
    const balanced = new Weighting(1, 1);
    const heavy = new Weighting(1, 9);

    expect(balanced.score(1, 0)).toBeCloseTo(1);
    expect(heavy.score(1, 0)).toBeCloseTo(1);
    expect(heavy.score(-1, 0)).toBeGreaterThanOrEqual(-1);
  });

  it('halves the recency contribution over the half-life', () => {
    const weighting = new Weighting(0, 1, 100);

    expect(weighting.decay(0)).toBeCloseTo(1);
    expect(weighting.decay(100)).toBeCloseTo(0.5);
    expect(weighting.decay(200)).toBeCloseTo(0.25);
  });

  it('DECAYS rather than cutting off', () => {
    // A cut-off shows up as an agent that knew something yesterday and does not
    // today, with nothing in between.
    const weighting = new Weighting(0, 1, 100);

    expect(weighting.decay(10_000)).toBeGreaterThan(0);
  });

  it('refuses weights that cannot rank anything', () => {
    expect(() => new Weighting(0, 0)).toThrowError(MemoryError);
    expect(() => new Weighting(-1, 2)).toThrowError(MemoryError);
    expect(() => new Weighting(1, 0, 0)).toThrowError(/half-life/);
  });
});

describe('the store', () => {
  it('reports itself VOLATILE', () => {
    expect(new InMemoryVectorStore().durability()).toBe('volatile');
  });

  it('upserts and searches by similarity', async () => {
    const store = new InMemoryVectorStore();
    await store.upsert([
      record({ id: 'a', vector: Vector.of([1, 0, 0]) }),
      record({ id: 'b', vector: Vector.of([0, 1, 0]) }),
    ]);

    const matches = await store.search({
      collections: ['default'],
      vector: Vector.of([1, 0, 0]),
      limit: 10,
    });

    expect(matches[0]?.record.id).toBe('a');
    expect(matches[0]?.similarity).toBeCloseTo(1);
  });

  it('REFUSES to mix two embedding spaces in one collection', async () => {
    // Vectors from two models are not comparable — the numbers are in different
    // spaces — and mixing them produces similarities that look plausible and
    // mean nothing.
    const store = new InMemoryVectorStore();
    await store.upsert([record({ id: 'a', embeddingModel: 'model-one' })]);

    await expect(
      store.upsert([record({ id: 'b', embeddingModel: 'model-two' })]),
    ).rejects.toMatchObject({ code: 'embedding_space_mismatch' });
  });

  it('skips records with no vector yet, and lists them for an embedder', async () => {
    const store = new InMemoryVectorStore();
    await store.upsert([record({ id: 'pending', vector: null, embeddingModel: null })]);

    expect(await store.search({ collections: ['default'], vector: Vector.of([1, 0, 0]), limit: 5 })).toEqual([]);
    expect((await store.unembedded('default')).map((r) => r.id)).toEqual(['pending']);
    expect(await store.count('default')).toBe(1);
    expect(await store.count('default', true)).toBe(0);
  });

  it('filters on metadata, and an array means ANY OF', async () => {
    const store = new InMemoryVectorStore();
    await store.upsert([
      record({ id: 'a', metadata: { topic: 'billing' } }),
      record({ id: 'b', metadata: { topic: 'shipping' } }),
      record({ id: 'c', metadata: { topic: 'returns' } }),
    ]);

    const one = await store.search({
      collections: ['default'],
      vector: Vector.of([1, 0, 0]),
      limit: 10,
      filter: { topic: 'billing' },
    });
    const many = await store.search({
      collections: ['default'],
      vector: Vector.of([1, 0, 0]),
      limit: 10,
      filter: { topic: ['billing', 'returns'] },
    });

    expect(one.map((m) => m.record.id)).toEqual(['a']);
    expect(many.map((m) => m.record.id).sort()).toEqual(['a', 'c']);
  });

  it('forgets, purges, and purges by observation time', async () => {
    const store = new InMemoryVectorStore();
    await store.upsert([
      record({ id: 'old', provenance: { source: null, author: null, observedAt: 100 } }),
      record({ id: 'new', provenance: { source: null, author: null, observedAt: 9999 } }),
    ]);

    expect(await store.purgeObservedBefore('default', 1000)).toBe(1);
    expect(await store.forget('default', ['new'])).toBe(1);
    expect(await store.count('default')).toBe(0);
    expect(await store.purge('default')).toBe(0);
  });

  it('searches several collections together', async () => {
    const store = new InMemoryVectorStore();
    await store.upsert([record({ id: 'a', collection: 'one' }), record({ id: 'b', collection: 'two' })]);

    const matches = await store.search({
      collections: ['one', 'two'],
      vector: Vector.of([1, 0, 0]),
      limit: 10,
    });

    expect(matches).toHaveLength(2);
  });
});

describe('recall', () => {
  it('OVER-FETCHES, so recency can reorder past the visible limit', async () => {
    // Rescoring the top 8 by similarity can only ever reorder those 8. A memory
    // that is the seventieth most similar and was written an hour ago cannot
    // win a recency-weighted ranking it was never entered into.
    const store = new InMemoryVectorStore();

    // 'stale' is the closest match; 'fresh' is further but much newer.
    await store.upsert([
      record({
        id: 'stale',
        vector: Vector.of([1, 0, 0]),
        provenance: { source: null, author: null, observedAt: 0 },
      }),
      record({
        id: 'fresh',
        vector: Vector.of([0.9, 0.4, 0]),
        provenance: { source: null, author: null, observedAt: 10_000 },
      }),
    ]);

    const byRelevance = await recall(
      store,
      { collections: ['default'], vector: Vector.of([1, 0, 0]) },
      new RecallSettings(1, 8, null, new Weighting(1, 0)),
      10_000,
    );
    const byRecency = await recall(
      store,
      { collections: ['default'], vector: Vector.of([1, 0, 0]) },
      new RecallSettings(1, 8, null, new Weighting(1, 3, 3600)),
      10_000,
    );

    expect(byRelevance.memories[0]?.record.id).toBe('stale');
    expect(byRecency.memories[0]?.record.id).toBe('fresh');
    // Both saw the same candidate pool — the ranking changed, not the fetch.
    expect(byRelevance.candidates).toBe(2);
  });

  it('asks the store for limit × overfetch candidates', async () => {
    let askedFor = 0;
    const store = new InMemoryVectorStore();
    const spy = {
      ...store,
      search: async (query: { limit: number }) => {
        askedFor = query.limit;

        return [];
      },
    };

    await recall(
      spy as never,
      { collections: ['default'], vector: Vector.of([1, 0, 0]) },
      new RecallSettings(4, 8),
    );

    expect(askedFor).toBe(32);
  });

  it('applies minScore AFTER reranking, not before', async () => {
    // The threshold is about the score the caller sees. Applying it to raw
    // similarity would filter on a number the caller never asked about.
    const store = new InMemoryVectorStore();
    await store.upsert([record({ id: 'a', vector: Vector.of([0.5, 0.5, 0]) })]);

    const strict = await recall(
      store,
      { collections: ['default'], vector: Vector.of([1, 0, 0]) },
      new RecallSettings(8, 8, 0.99),
      1000,
    );

    expect(strict.memories).toEqual([]);
    // It was still a candidate; it just did not clear the bar.
    expect(strict.candidates).toBe(1);
  });

  it('reports how many candidates it considered, for tuning overfetch', async () => {
    const store = new InMemoryVectorStore();
    await store.upsert([record({ id: 'a' }), record({ id: 'b' })]);

    const result = await recall(store, { collections: ['default'], vector: Vector.of([1, 0, 0]) });

    expect(result.candidates).toBe(2);
    expect(result.memories).toHaveLength(2);
  });
});
