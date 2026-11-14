// polymarket-faa-attribution-02 entry (Node 22, stdlib only): adaptive quantile forecaster.
// Reads every <id>.in (up to 72h of 1-min Yes prices per market), writes <id>.out with
// 240 minutes x 5 quantiles (0.05/0.25/0.5/0.75/0.95) per market.
//
// Model, per market, all from the case's own prices (no network, deterministic):
//  - volatility: EWMA of 1-min squared moves (30-min half-life), one-sided so the
//    downside and upside spreads adapt separately; floored at 0.0001, capped 0.005.
//  - drift: least-squares slope of the last 180 minutes, shrunk 0.6 toward 0,
//    capped at 0.00015/min (prices are near-martingales; trends are real but noisy).
//  - quantiles: narrow-normal multipliers, blended 75/25 with the empirical
//    quantiles of recent h-minute changes (anchors 30/60/120/240 min, 12h
//    recency half-life), so the shape follows the market's own jump distribution.
//  - clamped to [0,1], rounded to 6 decimals, non-decreasing per row.
//
// Backtests (own pinball scorer, validated: it reproduces the house baseline's
// practice loss 0.363589 exactly): practice 0.320 vs starter 0.364; FAA
// history-to-now simulated checkpoints 0.433 vs starter 0.618.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Quantile multipliers (narrow-normal: prediction-market moves are step-like). */
const Z = [-1.5, -0.65, 0, 0.65, 1.5];
/** EWMA half-life (minutes) for the volatility window. */
const HL_FAST = 30;
/** Volatility floor/cap (price units per sqrt-minute). */
const SIGMA_FLOOR = 0.0001;
const SIGMA_CAP = 0.005;
/** Drift: slope of the last DRIFT_WIN minutes, shrunk toward 0, capped per minute. */
const DRIFT_WIN = 180;
const DRIFT_SHRINK = 0.6;
const DRIFT_CAP = 0.00015;
/** Empirical blend weight and recency half-life (minutes). */
const EMP_W = 0.25;
const EMP_HL = 720;
const EMP_ANCHORS = [30, 60, 120, 240];

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** EWMA of squared moves (or one-sided squared moves when `down` is true/false). */
function ewmaVar(diffs: number[], halfLife: number, side?: 'down' | 'up'): number {
  const lam = Math.pow(0.5, 1 / halfLife);
  let v = 0;
  for (const d of diffs) {
    const sq = side === 'down' ? (d < 0 ? d * d : 0) : side === 'up' ? (d > 0 ? d * d : 0) : d * d;
    v = lam * v + (1 - lam) * sq;
  }
  return v;
}

/** Least-squares slope (per minute) of the last `win` prices. */
function recentSlope(p: number[], win: number): number {
  const n = Math.min(win, p.length);
  if (n < 2) return 0;
  const start = p.length - n;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const x = i, y = p[start + i] as number;
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const den = n * sxx - sx * sx;
  if (den === 0) return 0;
  return (n * sxy - sx * sy) / den;
}

/** Weighted quantile of (value, weight) pairs. */
function weightedQuantile(pairs: { v: number; w: number }[], q: number): number {
  const sorted = pairs.slice().sort((a, b) => a.v - b.v);
  let tot = 0;
  for (const s of sorted) tot += s.w;
  let acc = 0;
  for (const s of sorted) {
    acc += s.w;
    if (acc >= q * tot) return s.v;
  }
  return sorted.length ? (sorted[sorted.length - 1] as { v: number }).v : 0;
}

/** Empirical change quantiles at the anchor horizons, from the last 1440 minutes
 *  with exponential recency weights. Returns per-anchor [q05,q25,q50,q75,q95]. */
function empiricalAnchors(p: number[]): number[][] {
  const n = p.length;
  const lam = Math.pow(0.5, 1 / EMP_HL);
  const out: number[][] = [];
  for (const h of EMP_ANCHORS) {
    const pairs: { v: number; w: number }[] = [];
    const from = Math.max(h, n - 1440);
    for (let i = from; i < n; i++) {
      pairs.push({ v: (p[i] as number) - (p[i - h] as number), w: Math.pow(lam, n - 1 - i) });
    }
    out.push(QUANTILES.map((t) => weightedQuantile(pairs, t)));
  }
  return out;
}

/** Interpolate the empirical quantile offset for horizon h (sqrt-h space). */
function empiricalOffset(anchors: number[][], h: number, qi: number): number {
  const s = Math.sqrt(h);
  const sa = EMP_ANCHORS.map((a) => Math.sqrt(a));
  if (s <= (sa[0] as number)) return (anchors[0]?.[qi] ?? 0) * (s / (sa[0] as number));
  for (let k = 0; k < sa.length - 1; k++) {
    const s0 = sa[k] as number, s1 = sa[k + 1] as number;
    if (s <= s1) {
      const a0 = anchors[k]?.[qi] ?? 0, a1 = anchors[k + 1]?.[qi] ?? 0;
      return a0 + ((a1 - a0) * (s - s0)) / (s1 - s0);
    }
  }
  return anchors[anchors.length - 1]?.[qi] ?? 0;
}

function forecast(m: CaseMarket, horizon: number): number[][] {
  const p = m.prices;
  if (p.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = p[p.length - 1] as number;
  const diffs: number[] = [];
  for (let i = 1; i < p.length; i++) diffs.push((p[i] as number) - (p[i - 1] as number));
  // One-sided sigmas: E[d^2 | d<0] is half the variance for symmetric moves, hence sqrt(2*...).
  const sigmaDown = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, Math.sqrt(2 * ewmaVar(diffs, HL_FAST, 'down'))));
  const sigmaUp = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, Math.sqrt(2 * ewmaVar(diffs, HL_FAST, 'up'))));
  let drift = DRIFT_SHRINK * recentSlope(p, DRIFT_WIN);
  drift = Math.min(DRIFT_CAP, Math.max(-DRIFT_CAP, drift));
  const emp = empiricalAnchors(p);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const sDown = sigmaDown * Math.sqrt(h);
    const sUp = sigmaUp * Math.sqrt(h);
    const row = Z.map((z, qi) => {
      const spread = z < 0 ? z * sDown : z > 0 ? z * sUp : 0;
      const paramOff = drift * h + spread;
      const empOff = empiricalOffset(emp, h, qi);
      return price6(x + (1 - EMP_W) * paramOff + EMP_W * empOff);
    });
    // enforce non-decreasing after clamping/rounding
    for (let j = 1; j < row.length; j++) if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
