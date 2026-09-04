//! Full BaseFold multilinear opening: an eq-weighted sumcheck for
//! `ẑ(z) = Σ_b eq(z,b)·m[b]` **interleaved** with the codeword fold. Each
//! round's (random) sumcheck challenge is also the fold challenge, so the
//! proximity test folds with random challenges while the `eq(z,·)` weighting
//! pins the evaluation to the specific point `z`. After `d` rounds the folded
//! codeword is `base(ẑ(α))` and the sumcheck's final claim must equal
//! `eq(z,α)·ẑ(α)`, tying the recovered value back to the claimed `y`.

use super::super::brakedown::merkle::{Hash, MerkleTree, hash_leaf, verify_path};
use super::code::FoldableCode;
use super::query::{CommittedLayer, QueryOpening, commit_layer, leaf_bytes};
use crate::traits::PrimeFieldExt;
use ff::{Field, PrimeField};
use sha3::Shake256;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// A full multilinear-evaluation proof.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvalProof<F> {
  /// Per round: the quadratic sumcheck message as `(h(0), h(1), h(2))`.
  pub sumcheck: Vec<[F; 3]>,
  /// Merkle roots of the folded codewords `c¹, …, cᵈ`.
  pub layer_roots: Vec<Hash>,
  /// The base codeword `cᵈ`.
  pub base: Vec<F>,
  /// Fold-consistency query openings.
  pub queries: Vec<QueryOpening<F>>,
}

#[inline]
fn eqf<F: Field>(a: F, b: F) -> F {
  a * b + (F::ONE - a) * (F::ONE - b)
}

/// `eq(z, ·)` evaluations over the hypercube, `eq_table[idx] = Π_j eqf(z[j],
/// bit_j(idx))` — same index convention as the committed evaluation vector.
fn build_eq<F: Field>(z: &[F]) -> Vec<F> {
  let mut t = vec![F::ONE; 1usize << z.len()];
  for (j, &zj) in z.iter().enumerate() {
    let bit = 1usize << j;
    for (idx, ti) in t.iter_mut().enumerate() {
      let bj = if idx & bit != 0 { zj } else { F::ONE - zj };
      *ti *= bj;
    }
  }
  t
}

/// Evaluation-basis fold `(1−α)·lo + α·hi` (lo = first half = MSB 0).
fn fold_eval<F: Field>(t: &[F], alpha: F) -> Vec<F> {
  let half = t.len() / 2;
  (0..half)
    .map(|j| (F::ONE - alpha) * t[j] + alpha * t[half + j])
    .collect()
}

/// Evaluate the quadratic through `(0,h0),(1,h1),(2,h2)` at `x`.
fn quad_eval<F: Field>(h: &[F; 3], x: F) -> F {
  let inv2 = Option::<F>::from((F::ONE + F::ONE).invert()).expect("2 invertible");
  let one = F::ONE;
  let two = one + one;
  h[0] * (x - one) * (x - two) * inv2 - h[1] * x * (x - two) + h[2] * x * (x - one) * inv2
}

/// `eq(z, α)` with the sumcheck's MSB-first challenge ordering: round `j`
/// (challenge `α[j]`) bound variable `b_{d−j}`, i.e. `z`-coordinate `z[d−1−j]`.
fn eq_at_fold<F: Field>(z: &[F], alphas: &[F]) -> F {
  let d = z.len();
  alphas
    .iter()
    .enumerate()
    .fold(F::ONE, |acc, (j, &a)| acc * eqf(z[d - 1 - j], a))
}

fn absorb(x: &mut Shake256, label: &[u8], bytes: &[u8]) {
  x.update(&(label.len() as u64).to_le_bytes());
  x.update(label);
  x.update(&(bytes.len() as u64).to_le_bytes());
  x.update(bytes);
}

fn absorb_f<F: PrimeField>(x: &mut Shake256, label: &[u8], f: &F) {
  absorb(x, label, f.to_repr().as_ref());
}

fn squeeze_scalar<F: PrimeFieldExt>(x: &Shake256) -> F {
  let mut r = x.clone().finalize_xof();
  let mut b = [0u8; 64];
  r.read(&mut b);
  F::from_uniform(&b)
}

fn squeeze_index(x: &Shake256, ctr: u64, bound: usize) -> usize {
  let mut h = x.clone();
  h.update(b"idx");
  h.update(&ctr.to_le_bytes());
  let mut r = h.finalize_xof();
  let mut b = [0u8; 8];
  r.read(&mut b);
  (u64::from_le_bytes(b) % bound as u64) as usize
}

/// Prove `ẑ(z) = y` for the committed evaluations `m_evals` (`c0 =
/// commit(encode(m_evals))`).
pub fn prove_eval<F: PrimeFieldExt>(
  code: &FoldableCode<F>,
  c0: &CommittedLayer<F>,
  m_evals: &[F],
  z: &[F],
  y: F,
  n_queries: usize,
) -> EvalProof<F> {
  let d = code.log_k();
  assert_eq!(m_evals.len(), 1usize << d);
  assert_eq!(z.len(), d);

  let mut fs = Shake256::default();
  absorb(&mut fs, b"bf-c0", &c0.root);
  for zi in z {
    absorb_f(&mut fs, b"z", zi);
  }
  absorb_f(&mut fs, b"y", &y);

  let mut m = m_evals.to_vec();
  let mut eq = build_eq(z);
  let mut cur = c0.word.clone();
  let mut folded: Vec<CommittedLayer<F>> = Vec::with_capacity(d);
  let mut sumcheck: Vec<[F; 3]> = Vec::with_capacity(d);

  for i in 0..d {
    let half = m.len() / 2;
    let (mut h0, mut h1, mut h2) = (F::ZERO, F::ZERO, F::ZERO);
    for j in 0..half {
      h0 += eq[j] * m[j];
      h1 += eq[half + j] * m[half + j];
      let e2 = eq[half + j].double() - eq[j];
      let m2 = m[half + j].double() - m[j];
      h2 += e2 * m2;
    }
    absorb_f(&mut fs, b"h", &h0);
    absorb_f(&mut fs, b"h", &h1);
    absorb_f(&mut fs, b"h", &h2);
    let alpha: F = squeeze_scalar(&fs);
    sumcheck.push([h0, h1, h2]);

    m = fold_eval(&m, alpha);
    eq = fold_eval(&eq, alpha);
    cur = code.fold(&cur, alpha, d - i);
    let committed = commit_layer(cur.clone());
    absorb(&mut fs, b"cf", &committed.root);
    folded.push(committed);
  }

  let base = folded.last().expect("d>=1").word.clone();
  let layer_roots: Vec<Hash> = folded.iter().map(|l| l.root).collect();

  let word_at = |li: usize| -> &Vec<F> {
    if li == 0 {
      &c0.word
    } else {
      &folded[li - 1].word
    }
  };
  let tree_at = |li: usize| {
    if li == 0 {
      &c0.tree
    } else {
      &folded[li - 1].tree
    }
  };

  let n0 = code.codeword_len();
  let mut queries = Vec::with_capacity(n_queries);
  for q in 0..n_queries {
    let mut s = squeeze_index(&fs, q as u64, n0);
    let mut opened = Vec::with_capacity(d);
    let mut len = n0;
    for li in 0..d {
      let half = len / 2;
      let lo = s % half;
      let hi = lo + half;
      let w = word_at(li);
      let t = tree_at(li);
      opened.push((w[lo], t.path(lo), w[hi], t.path(hi)));
      s = lo;
      len = half;
    }
    queries.push(QueryOpening { layers: opened });
  }

  EvalProof {
    sumcheck,
    layer_roots,
    base,
    queries,
  }
}

/// Verify a `prove_eval` proof of `ẑ(z) = y` against commitment `c0_root`.
pub fn verify_eval<F: PrimeFieldExt>(
  code: &FoldableCode<F>,
  c0_root: &Hash,
  z: &[F],
  y: F,
  proof: &EvalProof<F>,
  n_queries: usize,
) -> bool {
  let d = code.log_k();
  if proof.sumcheck.len() != d
    || proof.layer_roots.len() != d
    || proof.queries.len() != n_queries
    || z.len() != d
  {
    return false;
  }
  // Base must be a genuine base codeword, matching its committed root.
  let Some(v) = code.base_value(&proof.base) else {
    return false;
  };
  let base_root = {
    let leaves: Vec<Hash> = proof
      .base
      .iter()
      .map(|x| hash_leaf(&leaf_bytes(x)))
      .collect();
    MerkleTree::from_leaves(leaves).root()
  };
  if base_root != proof.layer_roots[d - 1] {
    return false;
  }

  let mut fs = Shake256::default();
  absorb(&mut fs, b"bf-c0", c0_root);
  for zi in z {
    absorb_f(&mut fs, b"z", zi);
  }
  absorb_f(&mut fs, b"y", &y);

  let mut claim = y;
  let mut alphas: Vec<F> = Vec::with_capacity(d);
  for i in 0..d {
    let [h0, h1, h2] = proof.sumcheck[i];
    if h0 + h1 != claim {
      return false;
    }
    absorb_f(&mut fs, b"h", &h0);
    absorb_f(&mut fs, b"h", &h1);
    absorb_f(&mut fs, b"h", &h2);
    let alpha: F = squeeze_scalar(&fs);
    absorb(&mut fs, b"cf", &proof.layer_roots[i]);
    claim = quad_eval(&[h0, h1, h2], alpha);
    alphas.push(alpha);
  }

  // Final sumcheck claim ties the fold-recovered value v = ẑ(α) to y.
  if claim != eq_at_fold(z, &alphas) * v {
    return false;
  }

  // Proximity: the codeword fold chain (with the sumcheck challenges) is
  // consistent at random spot-checks.
  let roots: Vec<&Hash> = std::iter::once(c0_root)
    .chain(proof.layer_roots.iter())
    .collect();
  let n0 = code.codeword_len();
  for (q, query) in proof.queries.iter().enumerate() {
    if query.layers.len() != d {
      return false;
    }
    let mut s = squeeze_index(&fs, q as u64, n0);
    let mut chained: Option<F> = None;
    let mut len = n0;
    for (li, (vlo, plo, vhi, phi)) in query.layers.iter().enumerate() {
      let half = len / 2;
      let lo = s % half;
      let hi = lo + half;
      if !verify_path(roots[li], &leaf_bytes(vlo), lo, plo)
        || !verify_path(roots[li], &leaf_bytes(vhi), hi, phi)
      {
        return false;
      }
      let entry_at_s = if s < half { *vlo } else { *vhi };
      if let Some(exp) = chained
        && exp != entry_at_s
      {
        return false;
      }
      chained = Some(code.fold_pair(*vlo, *vhi, alphas[li], d - li, lo));
      s = lo;
      len = half;
    }
    if chained != Some(proof.base[s]) {
      return false;
    }
  }
  true
}

#[cfg(test)]
mod tests {
  use super::super::code::rand_vec;
  use super::*;
  use crate::provider::pt256::t256;

  type F = t256::Scalar;

  fn mle_eval(m: &[F], z: &[F]) -> F {
    build_eq(z).iter().zip(m).map(|(e, mi)| *e * *mi).sum()
  }

  #[test]
  fn eval_open_round_trip_and_tamper() {
    let d = 6;
    let c = 4;
    let nq = 24;
    let code = FoldableCode::<F>::new(d, c, b"eval-seed");
    let m: Vec<F> = rand_vec(1 << d, 5);
    let c0 = commit_layer(code.encode(&m));
    let z: Vec<F> = rand_vec(d, 8);
    let y = mle_eval(&m, &z);

    let proof = prove_eval(&code, &c0, &m, &z, y, nq);
    assert!(
      verify_eval(&code, &c0.root, &z, y, &proof, nq),
      "honest open verifies"
    );

    // Wrong claimed value -> reject.
    assert!(
      !verify_eval(&code, &c0.root, &z, y + F::ONE, &proof, nq),
      "wrong y rejected"
    );

    // Tamper a sumcheck message -> reject.
    let mut bad = proof.clone();
    bad.sumcheck[0][0] += F::ONE;
    assert!(
      !verify_eval(&code, &c0.root, &z, y, &bad, nq),
      "tampered sumcheck rejected"
    );

    // Tamper a query opening -> reject.
    let mut bad2 = proof.clone();
    bad2.queries[0].layers[0].0 += F::ONE;
    assert!(
      !verify_eval(&code, &c0.root, &z, y, &bad2, nq),
      "tampered opening rejected"
    );
  }
}
