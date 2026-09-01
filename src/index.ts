import { Buffer } from 'node:buffer';

export type MemoryErrorCode =
  /** A vector was built from values that cannot be scored. */
  | 'invalid_vector'
  /** Two vectors of different lengths were compared. */
  | 'dimension_mismatch'
  /** A record was written into a collection embedded by a different model. */
  | 'embedding_space_mismatch'
  /** A store that reports itself volatile was configured for durable memory. */
  | 'unsafe_memory_configuration'
  /** A record cannot be stored as given. */
  | 'unstorable_memory';

export class MemoryError extends Error {
  constructor(
    readonly code: MemoryErrorCode,
    message: string,
  ) {
    super(message);
    this.name = 'MemoryError';
  }
}

/**
 * What a stored record IS.
 *
 * ONE case, and that is a deliberate statement rather than an unfinished enum.
 * "What is stored — messages, summaries, or facts?" is the central open design
 * question in the reference's spec. Adding `summary` and `fact` before it is
 * answered would settle it QUIETLY: the cases would exist, something would
 * populate them, and the decision would have been made by whoever built first
 * rather than by anyone who weighed it.
 *
 * So this slice stores OBSERVATIONS — text that was actually said, with its
 * provenance — which is the substrate all three candidate answers share. Both
 * other cases land here additively later, and nothing stored today has to move.
 */
export type MemoryKind = 'observation';

/** Whether a store's contents survive a deploy. Same distinction as the harness. */
export type Durability = 'volatile' | 'durable';

// -- vectors -----------------------------------------------------------------

/**
 * An embedding, in a form that survives a database round trip UNCHANGED.
 *
 * The storage format is base64 of IEEE 754 doubles, explicitly LITTLE-ENDIAN —
 * byte for byte what the PHP reference's `pack('e*')` produces. Not JSON and
 * not float32, and both alternatives are rejected for reasons that are about
 * correctness rather than taste:
 *
 *  - **float32** loses roughly nine significant digits per component, so a
 *    vector written and read back scores differently against the same query
 *    than it did at write time. Nothing errors; the ranking just drifts, by an
 *    amount no test that only checks "did it come back" would notice.
 *  - **JSON** depends on the host's float formatting. A package whose stored
 *    numbers change because someone tuned a runtime setting is not storing
 *    numbers, it is storing opinions about them.
 *
 * Little-endian rather than machine order matters for the same reason: a row
 * written on one architecture and read on another would come back
 * byte-reversed. That is not hypothetical for a database, which is the one part
 * of a system that routinely outlives the machine that wrote to it.
 *
 * Matching the reference exactly is what lets a PHP app and a TypeScript or
 * Python one **read each other's rows**.
 *
 * The cost is size: 1536 dimensions is 12KB packed, 16KB base64. That is the
 * deliberate trade, and it is why a real vector database is the answer past a
 * certain scale rather than pretending this one is.
 */
export class Vector {
  #sumOfSquares: number | null = null;

  private constructor(readonly values: readonly number[]) {}

  /**
   * Build from untrusted numbers — a provider response, a config file, a test.
   *
   * VALIDATES EVERY COMPONENT. This is the write path and runs once per record,
   * so the O(n) pass is affordable and the guarantee is worth having: a single
   * NaN inside a stored vector makes every score computed against it NaN, and
   * NaN comparisons are false, so the record silently stops being retrievable
   * rather than failing.
   */
  static of(values: readonly (number | string)[]): Vector {
    if (values.length === 0) {
      throw new MemoryError('invalid_vector', 'A vector must have at least one component.');
    }

    const parsed: number[] = [];

    for (const [index, value] of values.entries()) {
      const component = typeof value === 'string' ? Number(value) : value;

      if (typeof component !== 'number' || !Number.isFinite(component)) {
        throw new MemoryError(
          'invalid_vector',
          `Component ${index} is not a finite number, so every score computed against this vector would be NaN.`,
        );
      }

      parsed.push(component);
    }

    const vector = new Vector(parsed);

    // Scorability is asserted HERE, on the write path, not lazily at the first
    // comparison. The reference does the same and the difference is not
    // cosmetic: a degenerate vector that is only rejected when something scores
    // against it has already been WRITTEN to a shared store, and every recall
    // that later touches that row throws instead of returning results. Failing
    // at construction puts the error where the caller can still do something
    // about it — it has the embedding, and it knows which document produced it.
    vector.sumOfSquares();

    return vector;
  }

  /** The stored form: base64 of little-endian float64s. */
  toStorage(): string {
    const buffer = Buffer.allocUnsafe(this.values.length * 8);

    for (const [index, value] of this.values.entries()) {
      buffer.writeDoubleLE(value, index * 8);
    }

    return buffer.toString('base64');
  }

  static fromStorage(encoded: string): Vector {
    const buffer = Buffer.from(encoded, 'base64');

    if (buffer.length === 0 || buffer.length % 8 !== 0) {
      throw new MemoryError(
        'invalid_vector',
        `A stored vector must be a whole number of 8-byte doubles; got ${buffer.length} byte(s).`,
      );
    }

    const values: number[] = [];

    for (let offset = 0; offset < buffer.length; offset += 8) {
      values.push(buffer.readDoubleLE(offset));
    }

    return Vector.of(values);
  }

  dimensions(): number {
    return this.values.length;
  }

  magnitude(): number {
    return Math.sqrt(this.sumOfSquares());
  }

  /**
   * Cosine similarity, clamped to [-1, 1].
   *
   * One square root rather than two: `dot / sqrt(a² · b²)` instead of
   * `dot / (sqrt(a²) · sqrt(b²))`. In the hot path one recall scores hundreds
   * of candidates against one query, and a square root is not free.
   */
  cosine(other: Vector): number {
    if (this.values.length !== other.values.length) {
      throw new MemoryError(
        'dimension_mismatch',
        `Cannot compare a ${this.dimensions()}-dimension vector with a ${other.dimensions()}-dimension one.`,
      );
    }

    let dot = 0;

    for (const [index, value] of this.values.entries()) {
      dot += value * other.values[index]!;
    }

    const product = this.sumOfSquares() * other.sumOfSquares();

    const similarity = Number.isFinite(product)
      ? dot / Math.sqrt(product)
      : // Two vectors large enough that the product of their squared lengths
        // overflows. Rarer than it sounds and not impossible, and the fallback
        // is exact enough — only the last bit is at stake, and Infinity would
        // cost every bit.
        dot / (this.magnitude() * other.magnitude());

    // Clamped so the RANGE is a guarantee even where the last bit is not.
    return Math.max(-1, Math.min(1, similarity));
  }

  /** Squared, because that is the form `cosine` wants. Cached: it is the hot path. */
  sumOfSquares(): number {
    if (this.#sumOfSquares !== null) return this.#sumOfSquares;

    let sum = 0;
    for (const value of this.values) sum += value * value;

    // Matches the reference's `assertScorable`, and the two cases are
    // genuinely different. A sum that OVERFLOWED cannot be scored at all; a
    // sum of ZERO is a vector with no direction. Collapsing them into one
    // message would send whoever reads it looking in the wrong place.
    if (!Number.isFinite(sum)) {
      throw new MemoryError(
        'invalid_vector',
        'This vector is too large to score: the sum of its squared components overflows a double.',
      );
    }

    if (sum <= 0) {
      throw new MemoryError(
        'invalid_vector',
        'A zero vector has no direction, so cosine similarity against it is undefined.',
      );
    }

    return (this.#sumOfSquares = sum);
  }
}

/**
 * A compact stand-in for a vector, so recall does not have to read the vectors.
 *
 * The bottleneck in a database-backed search is not the arithmetic, it is the
 * BYTES. Cosine over 1536 doubles is microseconds; the 12KB that vector occupies
 * takes far longer to get out of the database. Scoring ten thousand memories
 * means moving 120MB per recall, on every turn.
 *
 * So each vector also gets a signature: one bit per random hyperplane, recording
 * which side of it the vector falls on. Two vectors pointing in similar
 * directions agree on most bits, and the fraction they disagree on estimates the
 * angle between them directly — Hamming distance over `bits` is θ/π. At 256 bits
 * that is 32 bytes instead of 12KB, and it is enough to RANK.
 *
 * Ranking, not answering. The signature picks the candidates and the real
 * vectors decide the order, so every score a caller sees is an exact cosine
 * rather than an estimate. The approximation is confined to which memories got
 * considered.
 */
export class BinarySignature {
  readonly #planes: number[][];

  constructor(dimensions: number, bits = 256, seed = 0x5eed) {
    if (dimensions < 1 || bits < 1 || bits % 8 !== 0) {
      throw new MemoryError(
        'invalid_vector',
        'A signature needs at least one dimension and a whole number of bytes of bits.',
      );
    }

    // A DETERMINISTIC pseudo-random basis. The planes must be identical
    // everywhere the same collection is read, or two processes would compute
    // different signatures for the same vector and neither would be wrong —
    // the recall would simply stop finding things.
    let state = seed >>> 0;
    const next = (): number => {
      state = (state * 1664525 + 1013904223) >>> 0;

      return state / 0x1_0000_0000 - 0.5;
    };

    this.#planes = Array.from({ length: bits }, () =>
      Array.from({ length: dimensions }, () => next()),
    );
  }

  get bits(): number {
    return this.#planes.length;
  }

  /** One bit per hyperplane, packed into bytes and base64'd. */
  of(vector: Vector): string {
    const bytes = Buffer.alloc(this.#planes.length / 8);

    for (const [index, plane] of this.#planes.entries()) {
      let dot = 0;
      for (const [component, weight] of plane.entries()) dot += weight * (vector.values[component] ?? 0);

      if (dot >= 0) bytes[index >> 3]! |= 1 << (index % 8);
    }

    return bytes.toString('base64');
  }

  /** Hamming distance. Over `bits`, this estimates θ/π. */
  static distance(left: string, right: string): number {
    const a = Buffer.from(left, 'base64');
    const b = Buffer.from(right, 'base64');

    if (a.length !== b.length) {
      throw new MemoryError(
        'dimension_mismatch',
        'Two signatures of different lengths cannot be compared.',
      );
    }

    let distance = 0;

    for (const [index, byte] of a.entries()) {
      let xor = byte ^ b[index]!;
      while (xor !== 0) {
        distance += xor & 1;
        xor >>>= 1;
      }
    }

    return distance;
  }
}

// -- ranking -----------------------------------------------------------------

/**
 * How much relevance and how much recency.
 *
 * Pure similarity retrieves the most SIMILAR memory, which is not the same
 * thing as the most useful one. "What is my billing address?" is most similar
 * to every previous time the address was discussed — including the one from two
 * years ago that has since been superseded. Similarity has no opinion about
 * which of two matching memories is still true.
 *
 * Recency has the opposite failure: it retrieves the newest thing, related or
 * not. Neither axis is right alone, and which mix is right depends on what the
 * memory is FOR — a support history wants recency, a corpus of preferences
 * mostly does not. So the caller weights them, and the default is relevance
 * alone, because that is what someone expects from something called semantic
 * recall.
 *
 * Weights are NORMALISED, so a score stays inside [-1, 1] whatever the mix.
 * That is what keeps `minScore` meaning the same thing when the mix changes;
 * without it, raising the recency weight would raise every score and quietly
 * disable the caller's threshold.
 */
export class Weighting {
  static readonly DEFAULT_HALF_LIFE = 7 * 24 * 60 * 60;

  constructor(
    readonly relevance = 1,
    readonly recency = 0,
    /**
     * How long it takes a memory's recency contribution to halve.
     *
     * EXPONENTIAL rather than a cut-off. A cut-off makes a memory disappear the
     * moment it crosses a boundary, which shows up as an agent that knew
     * something yesterday and does not today, with nothing in between.
     */
    readonly halfLifeSeconds = Weighting.DEFAULT_HALF_LIFE,
  ) {
    if (relevance < 0 || recency < 0 || relevance + recency <= 0) {
      throw new MemoryError(
        'invalid_vector',
        'Weights cannot be negative and at least one must be above zero.',
      );
    }

    if (halfLifeSeconds <= 0) {
      throw new MemoryError('invalid_vector', 'A half-life must be a positive number of seconds.');
    }
  }

  decay(ageSeconds: number): number {
    if (ageSeconds <= 0) return 1;

    return 2 ** (-ageSeconds / this.halfLifeSeconds);
  }

  score(similarity: number, ageSeconds: number): number {
    const total = this.relevance + this.recency;

    return (similarity * this.relevance + this.decay(ageSeconds) * this.recency) / total;
  }

  usesRecency(): boolean {
    return this.recency > 0;
  }
}

/**
 * The defaults a recall uses when the caller does not say.
 *
 * `overfetch` is the one worth explaining. Ranking that considers anything
 * beyond raw similarity has to RESCORE, and rescoring the top 8 by similarity
 * can only ever reorder those 8 — a memory that is the seventieth most similar
 * and was written an hour ago cannot win a recency-weighted ranking it was
 * never entered into. So the store is asked for `limit × overfetch` candidates
 * and the weighting picks from those.
 *
 * Bigger is more faithful and more expensive, linearly, in the hot path. Eight
 * means a default recall of 8 considers 64.
 */
export class RecallSettings {
  constructor(
    readonly limit = 8,
    readonly overfetch = 8,
    readonly minScore: number | null = null,
    readonly weighting: Weighting = new Weighting(),
  ) {
    if (limit < 1 || overfetch < 1) {
      throw new MemoryError('invalid_vector', 'limit and overfetch must both be at least 1.');
    }
  }

  /** How many candidates the store is asked for. */
  candidateBudget(): number {
    return this.limit * this.overfetch;
  }
}

// -- records -----------------------------------------------------------------

export interface Provenance {
  /** Where this came from — a thread id, a document, a source system. */
  source: string | null;
  /** Who or what said it. */
  author: string | null;
  /** When it was said, as a unix timestamp in seconds. */
  observedAt: number | null;
}

export interface VectorRecord {
  id: string;
  collection: string;
  content: string;
  kind: MemoryKind;
  /** Null until something embeds it. See `unembedded()`. */
  vector: Vector | null;
  /** The model that produced the vector. Two spaces must never be mixed. */
  embeddingModel: string | null;
  metadata: Record<string, string | number | boolean | null>;
  provenance: Provenance;
  createdAt: number;
}

export interface VectorMatch {
  record: VectorRecord;
  /** The EXACT cosine, never a signature estimate. */
  similarity: number;
}

export interface Recalled extends VectorMatch {
  /** Similarity and recency, combined by the weighting. */
  score: number;
}

export interface Recollection {
  memories: readonly Recalled[];
  /** How many the store returned before reranking. Useful for tuning overfetch. */
  candidates: number;
}

export interface VectorQuery {
  collections: readonly string[];
  vector: Vector;
  /** What the STORE returns, not what the caller sees. See `RecallSettings`. */
  limit: number;
  /** Equality on metadata keys; an array means "any of". */
  filter?: Record<string, string | number | boolean | null | readonly (string | number | boolean)[]>;
  /** Applied by the store, before any reranking. */
  minSimilarity?: number | null;
}

export interface VectorStore {
  upsert(records: Iterable<VectorRecord>): Promise<void>;
  search(query: VectorQuery): Promise<VectorMatch[]>;
  forget(collection: string, recordIds: readonly string[]): Promise<number>;
  purge(collection: string): Promise<number>;
  purgeObservedBefore(collection: string, before: number): Promise<number>;
  count(collection: string, embeddedOnly?: boolean): Promise<number>;
  /** Records with no vector yet, for a background embedder to pick up. */
  unembedded(collection: string, limit?: number): Promise<VectorRecord[]>;
  durability(): Durability;
}

/**
 * A store in this process's memory. VOLATILE, and it says so.
 *
 * Right for a test or a single-process tool. A real deployment points at a
 * database or a vector database; this exists so the package WORKS ON INSTALL
 * rather than requiring infrastructure before the first memory can be written.
 */
export class InMemoryVectorStore implements VectorStore {
  readonly #records = new Map<string, VectorRecord>();

  durability(): Durability {
    return 'volatile';
  }

  async upsert(records: Iterable<VectorRecord>): Promise<void> {
    for (const record of records) {
      // The space guard. Two vectors from different models are not comparable
      // — the numbers are in different spaces — and mixing them produces
      // similarities that look plausible and mean nothing.
      const existing = [...this.#records.values()].find(
        (candidate) =>
          candidate.collection === record.collection &&
          candidate.embeddingModel !== null &&
          record.embeddingModel !== null &&
          candidate.embeddingModel !== record.embeddingModel,
      );

      if (existing !== undefined) {
        throw new MemoryError(
          'embedding_space_mismatch',
          `Collection [${record.collection}] is embedded by [${existing.embeddingModel}]; refusing to mix in [${record.embeddingModel}]. ` +
            'Vectors from two models are not comparable, and mixing them produces similarities that look plausible and mean nothing.',
        );
      }

      this.#records.set(`${record.collection} ${record.id}`, record);
    }
  }

  async search(query: VectorQuery): Promise<VectorMatch[]> {
    const matches: VectorMatch[] = [];

    for (const record of this.#records.values()) {
      if (!query.collections.includes(record.collection)) continue;
      if (record.vector === null) continue;
      if (!matchesFilter(record, query.filter)) continue;

      const similarity = query.vector.cosine(record.vector);

      if (query.minSimilarity != null && similarity < query.minSimilarity) continue;

      matches.push({ record, similarity });
    }

    matches.sort((a, b) => b.similarity - a.similarity);

    return matches.slice(0, query.limit);
  }

  async forget(collection: string, recordIds: readonly string[]): Promise<number> {
    let removed = 0;

    for (const id of recordIds) {
      if (this.#records.delete(`${collection} ${id}`)) removed += 1;
    }

    return removed;
  }

  async purge(collection: string): Promise<number> {
    return this.#removeWhere((record) => record.collection === collection);
  }

  async purgeObservedBefore(collection: string, before: number): Promise<number> {
    return this.#removeWhere(
      (record) =>
        record.collection === collection && (record.provenance.observedAt ?? record.createdAt) < before,
    );
  }

  async count(collection: string, embeddedOnly = false): Promise<number> {
    return [...this.#records.values()].filter(
      (record) => record.collection === collection && (!embeddedOnly || record.vector !== null),
    ).length;
  }

  async unembedded(collection: string, limit = 100): Promise<VectorRecord[]> {
    return [...this.#records.values()]
      .filter((record) => record.collection === collection && record.vector === null)
      .slice(0, limit);
  }

  #removeWhere(predicate: (record: VectorRecord) => boolean): number {
    let removed = 0;

    for (const [key, record] of [...this.#records.entries()]) {
      if (predicate(record)) {
        this.#records.delete(key);
        removed += 1;
      }
    }

    return removed;
  }
}

function matchesFilter(record: VectorRecord, filter: VectorQuery['filter']): boolean {
  if (filter === undefined) return true;

  for (const [key, wanted] of Object.entries(filter)) {
    const actual = record.metadata[key] ?? null;

    if (Array.isArray(wanted)) {
      if (!wanted.includes(actual as never)) return false;
      continue;
    }

    if (actual !== wanted) return false;
  }

  return true;
}

/**
 * Recall: over-fetch by similarity, then rescore with the weighting.
 *
 * The two-stage shape is the whole point — see `RecallSettings.overfetch`.
 */
export async function recall(
  store: VectorStore,
  query: Omit<VectorQuery, 'limit'>,
  settings: RecallSettings = new RecallSettings(),
  now: number = Math.floor(Date.now() / 1000),
): Promise<Recollection> {
  const candidates = await store.search({ ...query, limit: settings.candidateBudget() });

  const scored: Recalled[] = candidates.map((match) => ({
    ...match,
    score: settings.weighting.score(
      match.similarity,
      now - (match.record.provenance.observedAt ?? match.record.createdAt),
    ),
  }));

  scored.sort((a, b) => b.score - a.score);

  const kept =
    settings.minScore === null ? scored : scored.filter((match) => match.score >= settings.minScore!);

  return { memories: kept.slice(0, settings.limit), candidates: candidates.length };
}
