import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MemoryError, Vector } from '../src/index.js';

/**
 * The cross-language vector-storage corpus from `prism-parity`.
 *
 * A vector is written by whichever service embedded it and read back by
 * whichever service is recalling, so the bytes have to cross the language
 * boundary intact. A base64 string that decodes to different doubles elsewhere
 * does not error — it silently scores wrong, and a recall that returns the
 * wrong memory looks exactly like one that returned a mediocre one.
 *
 * This suite is `full` in all three languages, and it did not start that way:
 * it found that this port asserted scorability LAZILY, at the first comparison,
 * where the reference asserts it on the write path. Two rows disagreed. That
 * was fixed here rather than recorded as a divergence, because unlike G-20 and
 * G-21 it had a right answer and the reference had it. See G-22.
 */
interface StorageCase {
  id: string;
  title: string;
  values: number[];
  storage: {
    php: { refused: boolean; packed: string | null; round_trips: boolean | null };
    ts: { refused: boolean; packed: string | null; round_trips: boolean | null };
    py: { refused: boolean; packed: string | null; round_trips: boolean | null };
  };
  agrees: boolean;
  notes: string;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/memory-vector-storage.json', import.meta.url), 'utf8'),
) as { cases: StorageCase[] };

const storageOf = (entry: StorageCase) => {
  try {
    const vector = Vector.of(entry.values);
    const packed = vector.toStorage();

    return {
      refused: false,
      packed,
      round_trips:
        JSON.stringify(Vector.fromStorage(packed).values) === JSON.stringify(vector.values),
    };
  } catch (error) {
    if (!(error instanceof MemoryError)) throw error;

    return { refused: true, packed: null, round_trips: null };
  }
};

describe('the cross-language vector-storage corpus', () => {
  it('is the whole suite, not a subset someone trimmed to green', () => {
    expect(corpus.cases).toHaveLength(9);
  });

  it.each(corpus.cases)('$id stores exactly what the reference stores ($title)', (entry) => {
    expect(storageOf(entry)).toEqual(entry.storage.php);
  });

  it('agrees with the reference on EVERY row', () => {
    // Stated as its own assertion so the suite's `full` status is a thing the
    // test claims rather than a thing the manifest asserts about it.
    expect(corpus.cases.filter((entry) => !entry.agrees)).toEqual([]);
  });

  it('refuses a degenerate vector at the WRITE path, not at the first score', () => {
    // G-22. The difference is not cosmetic: a vector rejected only when
    // something scores against it has already been written to a shared store,
    // and every recall that later touches that row throws instead of returning
    // results. Failing here puts the error where the caller still has the
    // embedding and knows which document produced it.
    expect(() => Vector.of([0, -0])).toThrowError(/no direction/);
    expect(() => Vector.of([1e-300, 1e300])).toThrowError(/too large to score/);
  });

  it('round-trips every vector it accepts', () => {
    // Asserted separately from the byte comparison: a pack that no longer
    // unpacks to its own input is invisible to a test that only ever compares
    // packs to packs.
    for (const entry of corpus.cases.filter((row) => !row.storage.php.refused)) {
      expect(storageOf(entry).round_trips).toBe(true);
    }
  });
});
