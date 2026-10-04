/**
 * Seeded Pseudo-Random Number Generator (Mulberry32)
 * Produces deterministic, reproducible floating-point numbers in [0, 1).
 */
export function createMulberry32(seed: number = 123456789): () => number {
  let s = (seed >>> 0) || 1;
  return function next(): number {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
