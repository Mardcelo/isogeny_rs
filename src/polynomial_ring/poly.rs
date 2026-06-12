#![allow(dead_code)] // for now

use fp2::traits::Fp as FpTrait;
use rand_core::{CryptoRng, RngCore};

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

// With the dedicated kernels for products with operands of length <= 3,
// recursing all the way down beats schoolbook for everything above this.
const KARATSUBA_THRESHOLD: usize = 3;

/// Trait for arithmetic for univariate polynomials in Fp[X]
pub trait Poly<Fp: FpTrait>:
    Clone
    + Default
    + Display
    + Index<usize>
    + IndexMut<usize>
    + Sized
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + MulAssign<Self>
    + MulAssign<Fp>
{
    fn new_from_ele(a: &Fp) -> Self;
    fn new_from_slice(a: &[Fp]) -> Self;
    fn set_from_slice(&mut self, a: &[Fp]);

    fn degree(&self) -> Option<usize>;

    /// The coefficients of the polynomial as a slice, constant term first.
    fn coefficients(&self) -> &[Fp];

    fn reverse(&self) -> Self;

    fn scale(&self, a: &Fp) -> Self;

    fn evaluate(&self, a: &Fp) -> Fp;

    fn product_from_quadratic_leaves(leaves: &[[Fp; 3]]) -> Self;
    /// As `product_from_quadratic_leaves`, but with each leaf a palindrome
    /// [a, b, a], allowing faster self-reciprocal multiplication.
    fn product_from_palindromic_leaves(leaves: &[[Fp; 3]]) -> Self;
    fn resultant_from_roots(&self, ai: &[Fp]) -> Fp;
}

#[derive(Clone, Debug)]
pub struct Polynomial<Fp: FpTrait> {
    coeffs: Vec<Fp>,
}

impl<Fp: FpTrait> Polynomial<Fp> {
    /// Create a polynomial from a finite field element.
    pub fn new_from_ele(a: &Fp) -> Self {
        Self { coeffs: vec![*a] }
    }

    /// Create a polynomial from a slice of finite field elements.
    pub fn new_from_slice(a: &[Fp]) -> Self {
        Self { coeffs: a.to_vec() }
    }

    /// Set the coefficients of a polynomial from a slice
    pub fn set_from_slice(&mut self, a: &[Fp]) {
        self.coeffs.copy_from_slice(a)
    }

    /// The length of the polynomial. TODO: should we trim trailing zeros? If so, how often?
    fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Return the degree of the polynomial.
    // TODO: should we trim trailing zeros and mutate while computing the degree or do the following
    // and allow trailing zeros to remain.
    pub fn degree(&self) -> Option<usize> {
        let mut i = self.len() - 1;
        while i > 0 && self.coeffs[i].is_zero() == u32::MAX {
            i -= 1;
        }
        if i == 0 && self.coeffs[0].is_zero() == u32::MAX {
            None
        } else {
            Some(i)
        }
    }

    /// Return the constant coefficient of the polynomial
    pub fn constant_coefficient(&self) -> Option<Fp> {
        if self.len() == 0 {
            return None;
        }
        Some(self.coeffs[0])
    }

    /// Reverse the coefficients of self in place.
    fn reverse_into(&mut self) {
        self.coeffs.reverse();
    }

    /// Return the polynomial with coefficents reversed.
    pub fn reverse(&self) -> Self {
        let mut r = self.clone();
        r.reverse_into();
        r
    }

    /// Return 0xFFFFFFFF if self and other represent the same polynomial.
    /// Otherwise, return 0x00000000.
    pub fn equals(&self, other: &Self) -> u32 {
        // TODO: do I want this constant time?
        // eg: let mut equals = ct_u32_eq(self.len() as u32, other.len() as u32);
        if self.len() != other.len() {
            return 0;
        }

        let mut equals = u32::MAX;
        for i in 0..self.len() {
            equals &= self.coeffs[i].equals(&other[i]);
        }
        equals
    }

    /// Return 0xFFFFFFFF if self is zero, otherwise, return 0x00000000.
    pub fn is_zero(&self) -> u32 {
        let mut is_zero = u32::MAX;
        for i in 0..self.len() {
            is_zero &= self.coeffs[i].is_zero();
        }
        is_zero
    }

    /// Compute f <-- f + g, assumes that the length of f and g are the same.
    fn add_into(f: &mut [Fp], g: &[Fp]) {
        debug_assert!(f.len() >= g.len());
        for i in 0..g.len() {
            f[i] += g[i];
        }
    }

    /// Compute f <-- f - g, assumes that the length of f and g are the same.
    fn sub_into(f: &mut [Fp], g: &[Fp]) {
        debug_assert!(f.len() >= g.len());
        for i in 0..g.len() {
            f[i] -= g[i];
        }
    }

    /// Compute f * g with O(len(f) * len(g)) Fp multiplications using
    /// schoolbook multiplication. Assumes that fg has enough space for
    /// the result (len(f) + len(g) - 1).
    fn schoolbook_multiplication(fg: &mut [Fp], f: &[Fp], g: &[Fp]) {
        debug_assert!(fg.len() >= f.len() + g.len() - 1);

        for i in 0..f.len() {
            for j in 0..g.len() {
                if i == 0 || j + 1 == g.len() {
                    fg[i + j] = f[i] * g[j]
                } else {
                    fg[i + j] += f[i] * g[j]
                }
            }
        }
    }

    /// Compute f * g with ~O(n^1.5) Fp multiplications using Karastuba multiplication.
    /// Assumes that fg has enough space for the result (len(f) + len(g) - 1).
    fn karatsuba_multiplication(fg: &mut [Fp], f: &[Fp], g: &[Fp]) {
        debug_assert!(fg.len() >= f.len() + g.len() - 1);

        // Ensure that the degree of f is larger or equal to g (for balancing the split later)
        if f.len() < g.len() {
            Self::karatsuba_multiplication(fg, g, f);
            return;
        }

        // If g has length zero, then we set f * g to be zero.
        if g.is_empty() {
            for c in fg.iter_mut() {
                *c = Fp::ZERO;
            }
            return;
        }

        // If g has length one we simply scale all coefficients by g0.
        if g.len() == 1 {
            let g0 = g[0];
            for i in 0..f.len() {
                fg[i] = f[i] * g0;
            }
            return;
        }

        // When f is linear, then we also know g is linear and we can
        // save one multiplication compared to the naive method.
        if f.len() == 2 {
            let t1 = f[0] + f[1];
            let t2 = g[0] + g[1];
            fg[0] = f[0] * g[0];
            fg[2] = f[1] * g[1];
            fg[1] = t1 * t2;
            fg[1] -= fg[0];
            fg[1] -= fg[2];
            return;
        }

        // Dedicated kernels for quadratic f, following the SQIsign reference
        // implementation (poly-mul.c). A 3x2 product costs 5M instead of the
        // naive 6M and a 3x3 product costs 6M instead of 9M.
        if f.len() == 3 {
            if g.len() == 2 {
                let t0 = f[0] * g[0];
                let t2 = f[1] * g[1];
                let t1 = (f[0] + f[1]) * (g[0] + g[1]) - t0 - t2;
                fg[3] = f[2] * g[1];
                fg[2] = f[2] * g[0] + t2;
                fg[1] = t1;
                fg[0] = t0;
                return;
            }

            // f and g are both quadratic: Karatsuba on the top halves
            // combined with two mixed sums.
            let lo = f[1] * g[1];
            let hi = f[2] * g[2];
            let mid = (f[1] + f[2]) * (g[1] + g[2]) - lo - hi;
            let t1 = (f[0] + f[1]) * (g[0] + g[1]);
            let t2 = (f[0] + f[2]) * (g[0] + g[2]);
            let f0g0 = f[0] * g[0];
            fg[0] = f0g0;
            fg[1] = t1 - f0g0 - lo;
            fg[2] = t2 - f0g0 - hi + lo;
            fg[3] = mid;
            fg[4] = hi;
            return;
        }

        // For small degree f, g we use basic multiplication strategies with
        // O(n^2) operations.
        if f.len() <= KARATSUBA_THRESHOLD {
            Self::schoolbook_multiplication(fg, f, g);
            return;
        }

        // We are now at the point where f.len() >= 4 and we will split f into
        // a high and low part at floor(f.len() / 2) to perform a divide and
        // conquer strategy by Karastuba.
        let nf = f.len() / 2;
        let mf = f.len() - nf;

        // When g is particularly small we cannot split g into two halves, so we
        // have to modify Karatsuba to split f at floor(f.len() / 2) and then
        // multiply f_lo and f_hi by the whole of g.
        if g.len() <= nf {
            // We can compute f_lo * g directly into the first nf + g.len() - 1 elements
            // of the output.
            Self::karatsuba_multiplication(&mut fg[..nf + g.len() - 1], &f[..nf], g);

            // We then compute f_hi * g which will have length mf + g.len() - 1.
            // The bottom g.len() - 1 elements we need to add into fg while we can
            // directly copy the rest of the elements into the top of fg.
            let mut fg_hi = vec![Fp::ZERO; mf + g.len() - 1];
            Self::karatsuba_multiplication(&mut fg_hi, &f[nf..nf + mf], g);
            Self::add_into(&mut fg[nf..nf + g.len() - 1], &fg_hi[..g.len() - 1]);
            fg[nf + g.len() - 1..].copy_from_slice(&fg_hi[g.len() - 1..]);

            return;
        }

        // We are now at the point where we can split f = f_lo + x^nf * f_hi and
        // g = g_lo + x^nf * g_hi without an issue, with deg(f) >= deg(g).
        let mg = g.len() - nf;

        // The idea is that we now perform three calls to karatsuba_multiplication
        // on half-length inputs. Writing f * g = fg_lo + x^fn (fg_mid) + x^2*fn (fg_hi)
        // we need to compute:
        //
        // - fg_lo = f_lo * g_lo
        // - fg_mid = f_lo * g_hi + f_hi * g_lo
        //          = (f_lo + f_hi) * (g_lo + g_hi) - f_lo * g_lo - f_hi * g_hi
        // - fg_hi  = f_hi * g_hi
        //
        // Which means we need to compute only three multiplications: f_lo * g_lo,
        // f_hi * g_hi and (f_lo + f_hi) * (f_lo + f_hi).
        //
        // - fg_lo will have length 2*fn - 1 and fill the bottom of fg[..2*fn - 1].
        // - fg_hi will have length fm + gm - 1 and fill the top (without overlap)
        //   of fg[2*fn..]
        // - fg_mid will have length max(fn, fm) + max(fn, gm) - 1 and will fill
        //   from fg[fn..fn + fg_mid_len], to do this we will add the result into
        //   fg after the copies above.

        // The fg_lo and fg_hi parts of the computation are simple and we can
        // multiply straight into the output buffer.
        Self::karatsuba_multiplication(&mut fg[..nf + nf - 1], &f[..nf], &g[..nf]);
        Self::karatsuba_multiplication(&mut fg[nf + nf..], &f[nf..], &g[nf..]);

        // Now we compute f_lo + f_hi and g_lo + g_hi.

        // As nf is floor(len(f) / 2) then mf will either be nf or nf + 1, so we
        // can fit the sum into mf space and then add nf elements to it.
        // TODO: work with less allocations?
        let mut f_mid = f[nf..].to_vec();
        Self::add_into(&mut f_mid[..nf], &f[..nf]);

        // For g_lo + g_hi we need to be more careful, as len(g) <= len(f) we might
        // have that mg < nf.
        let mut g_mid = vec![Fp::ZERO; nf.max(mg)];
        if mg < nf {
            g_mid.copy_from_slice(&g[..nf]);
            Self::add_into(&mut g_mid[..mg], &g[nf..]);
        } else {
            g_mid.copy_from_slice(&g[nf..]);
            Self::add_into(&mut g_mid[..nf], &g[..nf]);
        }

        // Now we have both pieces, we compute their product into a temp buffer.
        let mut fg_mid = vec![Fp::ZERO; mf + nf.max(mg) - 1];
        Self::karatsuba_multiplication(&mut fg_mid, &f_mid, &g_mid);

        // We then compute (f_lo + f_hi) * (f_lo + f_hi) - f_lo * g_lo - f_hi * g_hi
        // with two subtractions.
        Self::sub_into(&mut fg_mid[..nf + nf - 1], &fg[..nf + nf - 1]);
        Self::sub_into(&mut fg_mid[..mf + mg - 1], &fg[nf + nf..]);

        // We now need to set the remaining window of fg which doesn't overlap with
        // the high and low pieces, and add in the last part. The dumb thing to do
        // is to zero out the single element which has yet to be copied over, but
        // we could save one addition by instead doing one copy at fg[nf + nf - 1]
        // and then add in two windows from fg[nf .. nf + nf - 1] and then fg[nf + nf ...].
        fg[nf + nf - 1] = Fp::ZERO;
        Self::add_into(&mut fg[nf..nf + mf + nf.max(mg) - 1], &fg_mid);
    }

    /// TODO: implement a scratch buffer for karatsuba_multiplication reuse
    fn mul_into(fg: &mut [Fp], f: &[Fp], g: &[Fp]) {
        assert!(f.len() + g.len() - 1 <= fg.len());
        Self::karatsuba_multiplication(fg, f, g)
    }

    /// TODO: testing only.
    pub fn basic_mul(&self, other: &Self) -> Self {
        let mut coeffs = vec![Fp::ZERO; self.len() + other.len() - 1];
        Self::schoolbook_multiplication(&mut coeffs, &self.coeffs, &other.coeffs);
        Self { coeffs }
    }

    /// TODO: testing only.
    pub fn karatsuba_mul(&self, other: &Self) -> Self {
        let mut coeffs = vec![Fp::ZERO; self.len() + other.len() - 1];
        Self::karatsuba_multiplication(&mut coeffs, &self.coeffs, &other.coeffs);
        Self { coeffs }
    }

    /// Set self to it's negative.
    fn set_neg(&mut self) {
        for x in self.coeffs.iter_mut() {
            x.set_neg();
        }
    }

    /// Set self <- self + other
    fn set_add(&mut self, other: &Self) {
        if self.len() < other.len() {
            self.coeffs.resize(other.len(), Fp::ZERO);
        }
        Self::add_into(&mut self.coeffs, &other.coeffs);
    }

    /// Set self <- self - other
    fn set_sub(&mut self, other: &Self) {
        if self.len() < other.len() {
            self.coeffs.resize(other.len(), Fp::ZERO);
        }
        Self::sub_into(&mut self.coeffs, &other.coeffs);
    }

    /// Set self <- self * other
    fn set_mul(&mut self, other: &Self) {
        let mut fg_coeffs = vec![Fp::ZERO; self.len() + other.len() - 1];
        Self::mul_into(&mut fg_coeffs, &self.coeffs, &other.coeffs);
        self.coeffs = fg_coeffs;
    }

    /// Multiply all coefficients of the polynomial by a element of the finite field.
    pub fn scale_into(&mut self, c: &Fp) {
        for x in self.coeffs.iter_mut() {
            *x *= *c;
        }
    }

    /// Return  c * self for some c in the finite field.
    pub fn scale(&self, c: &Fp) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.scale_into(c);
        r
    }

    /// Multiply all coefficients of the polynomial by a small value.
    pub fn scale_small_into(&mut self, k: i32) {
        for x in self.coeffs.iter_mut() {
            x.set_mul_small(k);
        }
    }

    /// Return c * self for some small c
    pub fn scale_small(&self, k: i32) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.scale_small_into(k);
        r
    }

    /// Computation of a product tree given an array of coefficients of
    /// quadratic polynomials as leaves. Avoids performing many memory
    /// allocations, but the code is a little harder to read.
    pub fn product_from_quadratic_leaves(leaves: &[[Fp; 3]]) -> Self {
        // Number of leaves and depth for the tree.
        let n = leaves.len();
        let log_n = usize::BITS - (2 * n - 1).leading_zeros();

        // Store the quadratic polynomials inside an input buffer
        let mut buf_in: Vec<Fp> = leaves
            .iter()
            .flat_map(|poly| poly.iter().copied())
            .collect();
        let mut buf_out = vec![Fp::ZERO; 3 * n];

        // Iterate over the remaining layers,
        for i in 1..log_n {
            // At each layer in the tree the degree of each polynomial is at most 2^i.
            let deg = 1 << i;
            let len = deg + 1;
            let out_len = 2 * deg + 1;

            // The majority of the multiplications will be h <- f * g where f and g
            // both have degree `deg`. This will be done k times when (2 * n = 2 * deg * k + r)
            let k = n / deg;
            let r = (n << 1) & ((1 << (i + 1)) - 1);

            // Keep track of the slices for the length deg + 1 polys for input and
            // length 2*degree + 1 polynomials as output for multiplication.
            let mut idx_in = 0;
            let mut idx_out = 0;

            // Compute k full multiplications into the buffer.
            for _ in 0..k {
                Self::mul_into(
                    &mut buf_out[idx_out..idx_out + out_len],
                    &buf_in[idx_in..idx_in + len],
                    &buf_in[idx_in + len..idx_in + 2 * len],
                );
                idx_out += out_len;
                idx_in += 2 * len;
            }

            if r > 0 {
                // By this point, we have consumed pairs of polynomials of degree `deg`
                // If `r` is larger than `deg` then we need to do an unbalanced multiplication
                // of a degree `deg` polynomial with a degree `r - deg` polynomial
                if r > deg {
                    let len_rem = r - deg + 1;
                    Self::mul_into(
                        &mut buf_out[idx_out..idx_out + r + 1],
                        &buf_in[idx_in..idx_in + len],
                        &buf_in[idx_in + len..idx_in + len + len_rem],
                    );
                } else {
                    // Otherwise we want to copy the polynomial from the in buffer to the
                    // out buffer
                    buf_out[idx_out..idx_out + r + 1]
                        .copy_from_slice(&buf_in[idx_in..idx_in + r + 1]);
                }
            }

            // Move the multiplication result into the input buffer and append
            // this into the layers.
            std::mem::swap(&mut buf_in, &mut buf_out);
        }

        Self::new_from_slice(&buf_in[..2 * n + 1])
    }

    /// UNUSED as we never use the full tree here.
    /// Computation of a product tree given an array of coefficients of
    /// quadratic polynomials as leaves. Avoids performing many memory
    /// allocations, but the code is a little harder to read.
    pub fn product_tree_quadratic_leaves(leaves: &[[Fp; 3]]) -> Vec<Vec<Fp>> {
        // Number of leaves and depth for the tree.
        let n = leaves.len();
        let log_n = usize::BITS - (2 * n - 1).leading_zeros();

        // Each layer of the tree is a vector of Fp elements, the leaves are n
        // polynomials of degree 2, then the next are n/2 polynomials of degree
        // 4 (and potenitally one of degree 2 if n is odd) and so on until the
        // root of the tree is a single polynomial of degree 2*n.
        let mut layers: Vec<Vec<Fp>> = Vec::with_capacity(log_n as usize);

        // Store the quadratic polynomials inside an input buffer
        let mut buf_in: Vec<Fp> = leaves
            .iter()
            .flat_map(|poly| poly.iter().copied())
            .collect();
        let mut buf_out = vec![Fp::ZERO; 3 * n];

        // Store the leaves of the tree in layers.
        layers.push(buf_in.clone());

        // Iterate over the remaining layers,
        for i in 1..log_n {
            // At each layer in the tree the degree of each polynomial is at most 2^i.
            let deg = 1 << i;
            let len = deg + 1;
            let out_len = 2 * deg + 1;

            // The majority of the multiplications will be h <- f * g where f and g
            // both have degree `deg`. This will be done k times when (2 * n = 2 * deg * k + r)
            let k = n / deg;
            let r = (n << 1) & ((1 << (i + 1)) - 1);

            // Keep track of the slices for the length deg + 1 polys for input and
            // length 2*degree + 1 polynomials as output for multiplication.
            let mut idx_in = 0;
            let mut idx_out = 0;

            // Compute k full multiplications into the buffer.
            for _ in 0..k {
                Self::mul_into(
                    &mut buf_out[idx_out..idx_out + out_len],
                    &buf_in[idx_in..idx_in + len],
                    &buf_in[idx_in + len..idx_in + 2 * len],
                );
                idx_out += out_len;
                idx_in += 2 * len;
            }

            if r > 0 {
                // By this point, we have consumed pairs of polynomials of degree `deg`
                // If `r` is larger than `deg` then we need to do an unbalanced multiplication
                // of a degree `deg` polynomial with a degree `r - deg` polynomial
                if r > deg {
                    let len_rem = r - deg + 1;
                    Self::mul_into(
                        &mut buf_out[idx_out..idx_out + r + 1],
                        &buf_in[idx_in..idx_in + len],
                        &buf_in[idx_in + len..idx_in + len + len_rem],
                    );
                } else {
                    // Otherwise we want to copy the polynomial from the in buffer to the
                    // out buffer
                    buf_out[idx_out..idx_out + r + 1]
                        .copy_from_slice(&buf_in[idx_in..idx_in + r + 1]);
                }
            }

            // Move the multiplication result into the input buffer and append
            // this into the layers.
            std::mem::swap(&mut buf_in, &mut buf_out);
            layers.push(buf_in.clone());
        }
        debug_assert!(layers.len() == log_n as usize);

        layers
    }

    /// Evaluate a polynomial at a value `a` using Horner's method.
    pub fn evaluate(&self, a: &Fp) -> Fp {
        // Handle degree 0 and 1 cases early.
        if self.len() == 0 {
            return Fp::ZERO;
        } else if self.len() == 1 {
            return self.coeffs[0];
        }

        // Iterate from the last to the first coefficicent using Horner's method
        // to evaluate the polynomial.
        let mut bi = Fp::ZERO;
        let deg = self.degree().unwrap();
        for i in 0..=deg {
            bi = *a * bi + self.coeffs[deg - i];
        }
        bi
    }

    /// Compute the resultant of self with a polynomial g = \prod {x - ai}
    /// given the roots ai.
    /// TODO: this is a naive method where we use Horner's eval on every root.
    /// Getting scaled remainder trees to be faster than this is a bit of a
    /// "white whale" problem for me, and is in the branch "remainder_tree"
    pub fn resultant_from_roots(&self, ai: &[Fp]) -> Fp {
        let mut res = Fp::ONE;
        for a in ai.iter() {
            res *= self.evaluate(a);
        }
        res
    }

    // ============================================================
    // Partial products (low and middle coefficients only), products
    // of self-reciprocal (palindromic) polynomials and projective
    // power series reciprocals.
    //
    // These are ports of the corresponding routines of the SQIsign
    // reference implementation (src/ec/ref/ecx/poly-mul.c and
    // poly-redc.c on the nist-v1 branch) and are the building blocks
    // for computing the resultants Res(hI, EJ) in sqrt-velu with
    // scaled remainder trees.
    //
    // All recursive functions take an explicit scratch buffer `tmp`
    // to avoid heap allocations in hot loops; the required scratch
    // length is given by the matching *_scratch_len function.

    /// Scratch length required by `mul_low_into` for output length n.
    pub fn mul_low_scratch_len(n: usize) -> usize {
        if n <= 3 {
            return 0;
        }
        let l0 = n - (n >> 1);
        // f0, f1, g0, g1 (combined at most 2n), f01 and g01 (at most l0
        // each), fg_0 (at most l0), fg_1 and fg_01 (at most n - l0 each),
        // plus the recursive calls which all have output length <= l0.
        2 * n + 3 * l0 + 2 * (n - l0) + Self::mul_low_scratch_len(l0)
    }

    /// Compute h[..n] <- f * g mod x^n, the lowest n coefficients of the
    /// product f * g. Entries of h past h[n - 1] are not touched. The
    /// scratch buffer must have length at least mul_low_scratch_len(n).
    fn mul_low_into(h: &mut [Fp], f: &[Fp], g: &[Fp], n: usize, tmp: &mut [Fp]) {
        debug_assert!(h.len() >= n);
        if n == 0 {
            return;
        }

        // Coefficients past x^(n-1) never contribute to the result, and we
        // keep f as the longer of the two inputs.
        let f = &f[..f.len().min(n)];
        let g = &g[..g.len().min(n)];
        let (f, g) = if f.len() >= g.len() { (f, g) } else { (g, f) };

        // Multiplication by the zero polynomial.
        if g.is_empty() {
            for c in h[..n].iter_mut() {
                *c = Fp::ZERO;
            }
            return;
        }

        // Multiplication mod x.
        if n == 1 {
            h[0] = f[0] * g[0];
            return;
        }

        // Case when no truncation is necessary.
        if n >= f.len() + g.len() - 1 {
            Self::karatsuba_multiplication(&mut h[..f.len() + g.len() - 1], f, g);
            for c in h[f.len() + g.len() - 1..n].iter_mut() {
                *c = Fp::ZERO;
            }
            return;
        }

        // From here on truncation is required, which forces g.len() >= 2.
        if n == 2 {
            let t0 = f[0] * g[1];
            let t1 = f[1] * g[0];
            h[0] = f[0] * g[0];
            h[1] = t0 + t1;
            return;
        }

        if n == 3 {
            if g.len() == 2 {
                let t1 = (f[0] + f[1]) * (g[0] + g[1]);
                let t2 = f[0] * g[0];
                let t0 = f[1] * g[1];
                h[2] = f[2] * g[0] + t0;
                h[1] = t1 - t2 - t0;
                h[0] = t2;
                return;
            }
            let t0 = f[0] * g[0];
            let t1 = f[1] * g[1];
            let t2 = (f[0] + f[1]) * (g[0] + g[1]) - t0 - t1;
            let t3 = (f[0] + f[2]) * (g[0] + g[2]);
            let t4 = f[2] * g[2];
            h[0] = t0;
            h[1] = t2;
            h[2] = t3 - t4 - t0 + t1;
            return;
        }

        // General case: split f and g into even and odd parts, e.g.
        // f(x) = f0(x^2) + x * f1(x^2), so that
        //     (f * g)(x) = fg_0(x^2) + x * fg_01(x^2) + x^2 * fg_1(x^2)
        // with
        //     fg_0  = f0 * g0                mod x^ceil(n / 2)
        //     fg_01 = f0 * g1 + f1 * g0      mod x^floor(n / 2)
        //     fg_1  = f1 * g1                mod x^(ceil(n / 2) - 1)
        // where the middle part is computed with a single multiplication
        // as (f0 + f1) * (g0 + g1) - fg_0 - fg_1.
        let l1 = n >> 1;
        let l0 = n - l1;

        let lenf1 = f.len() >> 1;
        let lenf0 = f.len() - lenf1;
        let leng1 = g.len() >> 1;
        let leng0 = g.len() - leng1;

        let (f0, rest) = tmp.split_at_mut(lenf0);
        let (f1, rest) = rest.split_at_mut(lenf1);
        let (g0, rest) = rest.split_at_mut(leng0);
        let (g1, rest) = rest.split_at_mut(leng1);
        let (f01, rest) = rest.split_at_mut(lenf0);
        let (g01, rest) = rest.split_at_mut(leng0);

        for i in 0..lenf1 {
            f0[i] = f[2 * i];
            f1[i] = f[2 * i + 1];
        }
        if lenf0 > lenf1 {
            f0[lenf0 - 1] = f[f.len() - 1];
        }
        for i in 0..leng1 {
            g0[i] = g[2 * i];
            g1[i] = g[2 * i + 1];
        }
        if leng0 > leng1 {
            g0[leng0 - 1] = g[g.len() - 1];
        }

        // f01 = f0 + f1 and g01 = g0 + g1.
        for i in 0..lenf1 {
            f01[i] = f0[i] + f1[i];
        }
        if lenf0 > lenf1 {
            f01[lenf0 - 1] = f0[lenf0 - 1];
        }
        for i in 0..leng1 {
            g01[i] = g0[i] + g1[i];
        }
        if leng0 > leng1 {
            g01[leng0 - 1] = g0[leng0 - 1];
        }

        let n0 = (lenf0 + leng0 - 1).min(l0);
        let n1 = (lenf1 + leng1 - 1).min(l1);
        let n01 = (lenf0 + leng0 - 1).min(l1);

        let (fg_0, rest) = rest.split_at_mut(n0);
        let (fg_1, rest) = rest.split_at_mut(n1);
        let (fg_01, rest) = rest.split_at_mut(n01);

        Self::mul_low_into(fg_0, f0, g0, n0, rest);
        Self::mul_low_into(fg_1, f1, g1, n1, rest);
        Self::mul_low_into(fg_01, f01, g01, n01, rest);

        // Subtract fg_0 and fg_1 from fg_01 to recover the cross term;
        // note that the lengths satisfy n1 <= n01 <= n0. The odd-index
        // coefficients of the result come from fg_01 and the even-index
        // ones interleave fg_0 and fg_1.
        let mut i = 0;
        while i < n1 {
            fg_01[i] -= fg_0[i];
            h[2 * i + 1] = fg_01[i] - fg_1[i];
            i += 1;
        }
        while i < n01 {
            h[2 * i + 1] = fg_01[i] - fg_0[i];
            i += 1;
        }

        h[0] = fg_0[0];
        for i in 1..n1 {
            h[2 * i] = fg_0[i] + fg_1[i - 1];
        }
        if 2 * n1 < n {
            h[2 * n1] = fg_0[n1] + fg_1[n1 - 1];
        }
    }

    /// Scratch length required by `quasi_mul_middle_into` for g of length m.
    pub fn quasi_mul_middle_scratch_len(m: usize) -> usize {
        if m <= 2 {
            return 0;
        }
        let l1 = m - (m >> 1);
        (2 * l1 - 1) + 2 * l1 + Self::quasi_mul_middle_scratch_len(l1)
    }

    /// Compute the middle part of the product f * g into h, specifically
    /// h[k] = (f * g)[m - 1 + k] for k in 0..m where m = g.len(). Requires
    /// f.len() >= 2 * m - 1; only the first 2 * m - 1 entries of f are used.
    ///
    /// This is the transposed-Karatsuba middle product from "The Middle
    /// Product Algorithm, I" by Hanrot, Quercia and Zimmermann; it costs as
    /// many multiplications as a Karatsuba product of half the size.
    fn quasi_mul_middle_into(h: &mut [Fp], g: &[Fp], f: &[Fp], tmp: &mut [Fp]) {
        let m = g.len();
        debug_assert!(h.len() >= m);
        if m == 0 {
            return;
        }
        debug_assert!(f.len() >= 2 * m - 1);

        if m == 1 {
            h[0] = f[0] * g[0];
            return;
        }

        // Unrolled recursion for the smallest case: 3M instead of the
        // naive 4M.
        if m == 2 {
            let a = (f[0] + f[1]) * g[1];
            let b = (g[1] - g[0]) * f[1];
            let c = (f[1] + f[2]) * g[0];
            h[0] = a - b;
            h[1] = c + b;
            return;
        }

        let l0 = m >> 1;
        let l1 = m - l0;
        let len_fsum = 2 * l1 - 1;

        let (fsum, rest) = tmp.split_at_mut(len_fsum);
        let (gbuf, rest) = rest.split_at_mut(l1);
        let (bbuf, rest) = rest.split_at_mut(l1);

        // A = middle(g_hi, f_lo + f_hi) into h[..l1].
        for i in 0..len_fsum {
            fsum[i] = f[i] + f[i + l1];
        }
        Self::quasi_mul_middle_into(&mut h[..l1], &g[l0..], fsum, rest);

        // B = middle(g_hi - g_lo, f_hi) into bbuf, where the subtraction is
        // shifted by one position when m is odd.
        if m & 1 == 1 {
            gbuf[0] = g[l0];
            for i in 0..l0 {
                gbuf[i + 1] = g[l1 + i] - g[i];
            }
        } else {
            for i in 0..l0 {
                gbuf[i] = g[l1 + i] - g[i];
            }
        }
        Self::quasi_mul_middle_into(bbuf, gbuf, &f[l1..], rest);

        // C = middle(g_lo, shifted sum of f) into h[l1..m].
        for i in 0..(2 * l0 - 1) {
            fsum[i] = f[i + l1] + f[i + 2 * l1];
        }
        Self::quasi_mul_middle_into(&mut h[l1..m], &g[..l0], &fsum[..2 * l0 - 1], rest);

        // Combine: h[..l1] = A - B and h[l1..] = C + B.
        for i in 0..l1 {
            h[i] -= bbuf[i];
        }
        for i in 0..l0 {
            h[l1 + i] += bbuf[i];
        }
    }

    /// Scratch length required by `mul_middle_into` for g of length leng
    /// and f of length lenf.
    pub fn mul_middle_scratch_len(leng: usize, lenf: usize) -> usize {
        if leng == 0 {
            return 0;
        }
        if leng == (lenf >> 1) && lenf & 1 == 0 {
            return lenf + Self::quasi_mul_middle_scratch_len(leng);
        }
        if leng == (lenf >> 1) + 1 && lenf & 1 == 0 {
            return lenf + 1 + Self::quasi_mul_middle_scratch_len(leng);
        }
        if leng == (lenf >> 1) + 1 && lenf & 1 == 1 {
            return Self::quasi_mul_middle_scratch_len(leng);
        }
        (lenf - leng) + leng + leng.saturating_sub(1) + Self::mul_low_scratch_len(leng)
    }

    /// Compute the middle part of the product f * g into h[..g.len()],
    /// specifically the coefficient window (f * g)[f.len() - g.len() ..
    /// f.len()). Requires f.len() >= g.len() > 0.
    fn mul_middle_into(h: &mut [Fp], g: &[Fp], f: &[Fp], tmp: &mut [Fp]) {
        let leng = g.len();
        let lenf = f.len();
        debug_assert!(lenf >= leng);
        if leng == 0 {
            return;
        }
        debug_assert!(h.len() >= leng);

        // deg(f) odd with deg(g) = floor(deg(f) / 2): pad f on the right so
        // that the window aligns with the quasi middle product.
        if leng == (lenf >> 1) && lenf & 1 == 0 {
            let (fpad, rest) = tmp.split_at_mut(lenf);
            fpad[..lenf - 1].copy_from_slice(&f[1..]);
            fpad[lenf - 1] = Fp::ZERO;
            Self::quasi_mul_middle_into(h, g, fpad, rest);
            return;
        }

        // deg(f) odd with deg(g) = ceil(deg(f) / 2): pad f on the left.
        if leng == (lenf >> 1) + 1 && lenf & 1 == 0 {
            let (fpad, rest) = tmp.split_at_mut(lenf + 1);
            fpad[0] = Fp::ZERO;
            fpad[1..].copy_from_slice(f);
            Self::quasi_mul_middle_into(h, g, fpad, rest);
            return;
        }

        // deg(f) even with deg(g) = deg(f) / 2: directly the quasi case.
        if leng == (lenf >> 1) + 1 && lenf & 1 == 1 {
            Self::quasi_mul_middle_into(h, g, f, tmp);
            return;
        }

        // Unbalanced cases: the window splits into the high part of
        // f_lo * g, computed as a low product of the reversals, plus the
        // low part of f_hi * g.
        let lenf0 = lenf - leng;
        let (f0rev, rest) = tmp.split_at_mut(lenf0);
        let (grev, rest) = rest.split_at_mut(leng);
        let (fg_low, rest) = rest.split_at_mut(leng.saturating_sub(1));

        for i in 0..lenf0 {
            f0rev[i] = f[lenf0 - 1 - i];
        }
        for i in 0..leng {
            grev[i] = g[leng - 1 - i];
        }

        Self::mul_low_into(fg_low, f0rev, grev, leng - 1, rest);
        Self::mul_low_into(h, &f[lenf0..], g, leng, rest);
        for i in 0..leng - 1 {
            h[i] += fg_low[leng - 2 - i];
        }
    }

    /// Scratch length required by `mul_selfreciprocal_into` for input
    /// lengths lenf and leng.
    pub fn mul_selfreciprocal_scratch_len(lenf: usize, leng: usize) -> usize {
        if lenf != leng {
            let m = (lenf + leng) >> 1;
            return Self::mul_low_scratch_len(m);
        }
        if lenf <= 5 {
            return 0;
        }
        if lenf & 1 == 1 {
            let len1 = lenf >> 1;
            let len0 = len1 + 1;
            // f0, g0, f1, g1, h0, h1 and h01 buffers plus recursion.
            2 * len0
                + 2 * len1
                + (2 * len0 - 1)
                + (2 * len1 - 1)
                + (2 * len0 - 1)
                + Self::mul_selfreciprocal_scratch_len(len0, len0)
                    .max(Self::mul_selfreciprocal_scratch_len(len1, len1))
        } else {
            // Even length: one extra buffer; the two half-size products go
            // through karatsuba_multiplication which manages its own memory.
            lenf - 1
        }
    }

    /// Compute h <- f * g for self-reciprocal f and g (their coefficient
    /// lists are palindromes; equivalently f(x) = x^deg(f) * f(1/x)).
    /// Writes f.len() + g.len() - 1 coefficients to h.
    fn mul_selfreciprocal_into(h: &mut [Fp], g: &[Fp], f: &[Fp], tmp: &mut [Fp]) {
        let lenf = f.len();
        let leng = g.len();
        debug_assert!(lenf == 0 || leng == 0 || h.len() >= lenf + leng - 1);

        // Different lengths: the product is again a palindrome, so compute
        // the low half with a short product and mirror the remainder.
        if lenf != leng {
            let m = (lenf + leng) >> 1;
            Self::mul_low_into(h, g, f, m, tmp);
            for i in m..lenf + leng - 1 {
                h[i] = h[lenf + leng - 2 - i];
            }
            return;
        }

        if lenf == 0 {
            return;
        }
        if lenf == 1 {
            h[0] = f[0] * g[0];
            return;
        }
        if lenf == 2 {
            h[0] = f[0] * g[0];
            h[1] = h[0].mul2();
            h[2] = h[0];
            return;
        }
        if lenf == 3 {
            let t1 = (g[0] + g[1]) * (f[0] + f[1]);
            let t0 = g[0] * f[0];
            let t2 = g[1] * f[1] + t0;
            h[0] = t0;
            h[1] = t1 - t2;
            h[2] = t2 + t0;
            h[3] = h[1];
            h[4] = h[0];
            return;
        }
        if lenf == 4 {
            let t1 = (g[0] + g[1]) * (f[0] + f[1]);
            let t0 = g[0] * f[0];
            let t3 = g[1] * f[1];
            let t2 = t1 - t0;
            h[0] = t0;
            h[1] = t2 - t3;
            h[2] = t2;
            h[3] = (t3 + t0).mul2();
            h[4] = t2;
            h[5] = h[1];
            h[6] = t0;
            return;
        }
        if lenf == 5 {
            let t1 = (g[1] - g[0]) * (f[0] - f[1]);
            let t2 = (g[2] - g[0]) * (f[0] - f[2]);
            let t3 = (g[2] - g[1]) * (f[1] - f[2]);
            let t0 = g[1] * f[1];
            let t4 = g[2] * f[2];
            h[0] = g[0] * f[0];
            let s = t0 + h[0];
            h[1] = t1 + s;
            h[3] = t3 + h[1] + t0 + t4;
            h[4] = s.mul2() + t4;
            h[2] = t2 + s + t4;
            h[5] = h[3];
            h[6] = h[2];
            h[7] = h[1];
            h[8] = h[0];
            return;
        }

        if lenf & 1 == 1 {
            // Odd equal lengths: split into even and odd parts, which are
            // themselves palindromes, and recurse three times. The cross
            // term is recovered from (f0 + (1+x) f1) * (g0 + (1+x) g1)
            // which is again a product of palindromes.
            let len1 = lenf >> 1;
            let len0 = len1 + 1;

            let (f0, rest) = tmp.split_at_mut(len0);
            let (g0, rest) = rest.split_at_mut(len0);
            let (f1, rest) = rest.split_at_mut(len1);
            let (g1, rest) = rest.split_at_mut(len1);
            let (h0, rest) = rest.split_at_mut(2 * len0 - 1);
            let (h1, rest) = rest.split_at_mut(2 * len1 - 1);
            let (h01, rest) = rest.split_at_mut(2 * len0 - 1);

            for i in 0..len1 {
                f0[i] = f[2 * i];
                g0[i] = g[2 * i];
                f1[i] = f[2 * i + 1];
                g1[i] = g[2 * i + 1];
            }
            f0[len1] = f[2 * len1];
            g0[len1] = g[2 * len1];

            Self::mul_selfreciprocal_into(h0, g0, f0, rest);
            Self::mul_selfreciprocal_into(h1, g1, f1, rest);

            for i in 0..len1 {
                g0[i] += g1[i];
                f0[i] += f1[i];
                g0[i + 1] += g1[i];
                f0[i + 1] += f1[i];
            }
            Self::mul_selfreciprocal_into(h01, g0, f0, rest);

            // h01 <- h01 - h0 - (1 + x)^2 h1.
            for i in 0..2 * len0 - 1 {
                h01[i] -= h0[i];
            }
            for i in 0..2 * len1 - 1 {
                h01[i] -= h1[i];
                h01[i + 1] -= h1[i].mul2();
                h01[i + 2] -= h1[i];
            }

            // Odd-index coefficients of h: divide h01 by (1 + x).
            h[1] = h01[0];
            for i in 1..2 * len0 - 2 {
                h[2 * i + 1] = h01[i] - h[2 * i - 1];
            }

            // Even-index coefficients interleave h0 and h1.
            h[0] = h0[0];
            for i in 1..2 * len1 {
                h[2 * i] = h0[i] + h1[i - 1];
            }
            h[4 * len1] = h0[2 * len1];
            return;
        }

        // Even equal lengths: two half-size general products; the top of
        // the result mirrors the bottom.
        let half = leng >> 1;
        let (h1, _) = tmp.split_at_mut(leng - 1);
        Self::karatsuba_multiplication(&mut h[..leng - 1], &g[..half], &f[..half]);
        Self::karatsuba_multiplication(h1, &g[..half], &f[half..leng]);
        h[leng - 1] = Fp::ZERO;
        for i in 0..half {
            h[half + i] += h1[i] + h1[leng - 2 - i];
        }
        for i in 0..leng - 1 {
            h[2 * leng - 2 - i] = h[i];
        }
    }

    /// Scratch length required by `reciprocal_into` for precision n.
    pub fn reciprocal_scratch_len(n: usize) -> usize {
        if n <= 4 {
            return 0;
        }
        let m = n - (n >> 1);
        2 * m
            + (n - m)
            + Self::reciprocal_scratch_len(m)
                .max(Self::mul_middle_scratch_len(m, n))
                .max(Self::mul_low_scratch_len(n - m))
    }

    /// Compute h[..n] and a constant c such that f * h = c mod x^n. The
    /// reciprocal is projective: no field inversions are performed.
    /// Requires f.len() >= n and h.len() >= n. Returns c.
    fn reciprocal_into(h: &mut [Fp], f: &[Fp], n: usize, tmp: &mut [Fp]) -> Fp {
        debug_assert!(f.len() >= n);
        debug_assert!(h.len() >= n);

        if n == 0 {
            return Fp::ZERO;
        }
        if n == 1 {
            h[0] = Fp::ONE;
            return f[0];
        }
        if n == 2 {
            h[0] = f[0];
            h[1] = -f[1];
            return f[0].square();
        }
        if n == 3 {
            let t0 = f[1].square();
            let mut t1 = f[0] * f[2];
            t1 -= t0;
            t1 *= f[0];
            let c = f[0].square();
            h[0] = f[0] * c;
            h[1] = -(f[1] * c);
            h[2] = -t1;
            return c.square();
        }
        if n == 4 {
            // Lift the order-2 reciprocal (f[0], -f[1]) with constant
            // f[0]^2 to order 4 directly.
            let g0 = f[0];
            let g1 = -f[1];
            let c2 = f[0].square();
            let t0 = f[1].square();
            let t1 = g0 * f[2];
            let s0 = t1 - t0;
            let s1 = g0 * f[3] + g1 * f[2];
            let u2 = s0 * g0;
            let u3 = s0 * g1 + s1 * g0;
            h[0] = g0 * c2;
            h[1] = g1 * c2;
            h[2] = -u2;
            h[3] = -u3;
            return c2.square();
        }

        // Newton iteration: lift a reciprocal mod x^m for m = ceil(n / 2)
        // to one mod x^n using one middle and one low product:
        // f * g - c is divisible by x^m, so only the terms in the window
        // [m, n) need to be corrected.
        let m = n - (n >> 1);
        let (g, rest) = tmp.split_at_mut(m);
        let (t, rest) = rest.split_at_mut(m);
        let (t2, rest) = rest.split_at_mut(n - m);

        let c = Self::reciprocal_into(g, f, m, rest);
        Self::mul_middle_into(t, g, &f[..n], rest);
        Self::mul_low_into(t2, g, &t[2 * m - n..], n - m, rest);
        for i in 0..m {
            h[i] = g[i] * c;
        }
        for i in m..n {
            h[i] = -t2[i - m];
        }
        c.square()
    }

    /// Scratch length required by `reduce_mod_scaled_into` for g of length
    /// leng reduced modulo f of length lenf.
    pub fn reduce_mod_scratch_len(leng: usize, lenf: usize) -> usize {
        if leng < lenf {
            return 0;
        }
        let lenq = leng - lenf + 1;
        3 * lenq + Self::mul_low_scratch_len(lenq).max(Self::mul_low_scratch_len(lenf - 1))
    }

    /// Compute h[..f.len() - 1] <- a * (g mod f) for a fixed scalar a which
    /// depends only on f and the *length* of g. Requires the precomputed
    /// projective reciprocal f_rev_inv of the reversal of f, satisfying
    /// rev(f) * f_rev_inv = c mod x^(g.len() - f.len() + 1).
    ///
    /// Since the scalar a only depends on f and g.len(), ratios of the
    /// outputs for equal-length g are exact, which is all sqrt-velu needs.
    fn reduce_mod_scaled_into(
        h: &mut [Fp],
        g: &[Fp],
        f: &[Fp],
        f_rev_inv: &[Fp],
        c: &Fp,
        tmp: &mut [Fp],
    ) {
        let leng = g.len();
        let lenf = f.len();
        debug_assert!(h.len() >= lenf - 1);

        // No reduction required.
        if leng < lenf {
            h[..leng].copy_from_slice(g);
            for x in h[leng..lenf - 1].iter_mut() {
                *x = Fp::ZERO;
            }
            return;
        }

        let lenq = leng - lenf + 1;
        debug_assert!(f_rev_inv.len() >= lenq);
        let (grev, rest) = tmp.split_at_mut(lenq);
        let (q, rest) = rest.split_at_mut(lenq);
        let (qrev, rest) = rest.split_at_mut(lenq);

        // The reversed quotient: q = rev(g) * f_rev_inv mod x^lenq. Only
        // the top lenq coefficients of g contribute.
        for i in 0..lenq {
            grev[i] = g[leng - 1 - i];
        }
        Self::mul_low_into(q, f_rev_inv, grev, lenq, rest);
        for i in 0..lenq {
            qrev[i] = q[lenq - 1 - i];
        }

        // h = c * g - quotient * f mod x^(lenf - 1). When f is monic the
        // reciprocal constant is exactly one and the scaling is free.
        Self::mul_low_into(h, qrev, f, lenf - 1, rest);
        if c.equals(&Fp::ONE) == u32::MAX {
            for i in 0..lenf - 1 {
                h[i] = g[i] - h[i];
            }
        } else {
            for i in 0..lenf - 1 {
                h[i] = *c * g[i] - h[i];
            }
        }
    }

    /// Compute self * other mod x^n as a new polynomial.
    pub fn mul_low(&self, other: &Self, n: usize) -> Self {
        let mut coeffs = vec![Fp::ZERO; n];
        let mut tmp = vec![Fp::ZERO; Self::mul_low_scratch_len(n)];
        Self::mul_low_into(&mut coeffs, &self.coeffs, &other.coeffs, n, &mut tmp);
        Self { coeffs }
    }

    /// Compute the middle part of the product self * other, that is the
    /// coefficient window [len(self) - len(other), len(self)), as a new
    /// polynomial. Requires len(self) >= len(other).
    pub fn mul_middle(&self, other: &Self) -> Self {
        let leng = other.len();
        let mut coeffs = vec![Fp::ZERO; leng];
        let mut tmp = vec![Fp::ZERO; Self::mul_middle_scratch_len(leng, self.len())];
        Self::mul_middle_into(&mut coeffs, &other.coeffs, &self.coeffs, &mut tmp);
        Self { coeffs }
    }

    /// Compute self * other as a new polynomial, where both self and other
    /// are self-reciprocal (palindromic coefficients).
    pub fn mul_selfreciprocal(&self, other: &Self) -> Self {
        let mut coeffs = vec![Fp::ZERO; self.len() + other.len() - 1];
        let mut tmp = vec![Fp::ZERO; Self::mul_selfreciprocal_scratch_len(self.len(), other.len())];
        Self::mul_selfreciprocal_into(&mut coeffs, &other.coeffs, &self.coeffs, &mut tmp);
        Self { coeffs }
    }

    /// Compute the projective reciprocal of self to precision n: a
    /// polynomial h and constant c such that self * h = c mod x^n.
    /// Requires len(self) >= n.
    pub fn reciprocal(&self, n: usize) -> (Self, Fp) {
        let mut coeffs = vec![Fp::ZERO; n];
        let mut tmp = vec![Fp::ZERO; Self::reciprocal_scratch_len(n)];
        let c = Self::reciprocal_into(&mut coeffs, &self.coeffs, n, &mut tmp);
        (Self { coeffs }, c)
    }

    /// Computation of the product of palindromic quadratic polynomials.
    /// Identical in structure to `product_from_quadratic_leaves` but uses
    /// the specialised self-reciprocal multiplication: every node of the
    /// implicit product tree is itself a palindrome, which makes each
    /// multiplication cost roughly two thirds of the general method.
    pub fn product_from_palindromic_quadratic_leaves(leaves: &[[Fp; 3]]) -> Self {
        let n = leaves.len();
        let log_n = usize::BITS - (2 * n - 1).leading_zeros();

        let mut buf_in: Vec<Fp> = leaves
            .iter()
            .flat_map(|poly| poly.iter().copied())
            .collect();
        let mut buf_out = vec![Fp::ZERO; 3 * n];

        // One reusable scratch buffer sized for the largest multiplication
        // in the tree.
        let mut scratch_len = 0;
        for i in 1..log_n {
            let deg = 1 << i;
            let len = deg + 1;
            let r = (n << 1) & ((1 << (i + 1)) - 1);
            scratch_len = scratch_len.max(Self::mul_selfreciprocal_scratch_len(len, len));
            if r > deg {
                scratch_len =
                    scratch_len.max(Self::mul_selfreciprocal_scratch_len(len, r - deg + 1));
            }
        }
        let mut scratch = vec![Fp::ZERO; scratch_len];

        for i in 1..log_n {
            let deg = 1 << i;
            let len = deg + 1;
            let out_len = 2 * deg + 1;

            let k = n / deg;
            let r = (n << 1) & ((1 << (i + 1)) - 1);

            let mut idx_in = 0;
            let mut idx_out = 0;

            for _ in 0..k {
                Self::mul_selfreciprocal_into(
                    &mut buf_out[idx_out..idx_out + out_len],
                    &buf_in[idx_in..idx_in + len],
                    &buf_in[idx_in + len..idx_in + 2 * len],
                    &mut scratch,
                );
                idx_out += out_len;
                idx_in += 2 * len;
            }

            if r > 0 {
                if r > deg {
                    let len_rem = r - deg + 1;
                    Self::mul_selfreciprocal_into(
                        &mut buf_out[idx_out..idx_out + r + 1],
                        &buf_in[idx_in..idx_in + len],
                        &buf_in[idx_in + len..idx_in + len + len_rem],
                        &mut scratch,
                    );
                } else {
                    buf_out[idx_out..idx_out + r + 1]
                        .copy_from_slice(&buf_in[idx_in..idx_in + r + 1]);
                }
            }

            std::mem::swap(&mut buf_in, &mut buf_out);
        }

        Self::new_from_slice(&buf_in[..2 * n + 1])
    }

    /// Set self to a random value
    pub fn set_rand<R: CryptoRng + RngCore>(&mut self, rng: &mut R) {
        for x in self.coeffs.iter_mut() {
            x.set_rand(rng);
        }
    }

    /// Return a new random polynomial with length d
    pub fn rand<R: CryptoRng + RngCore>(rng: &mut R, d: usize) -> Self {
        let mut r = Self {
            coeffs: vec![Fp::ZERO; d],
        };
        r.set_rand(rng);
        r
    }
}

/// A product tree over a set of roots a_i together with a precomputed
/// projective reciprocal of (the reversal of) its root polynomial, used to
/// repeatedly compute scaled resultants Res(prod_i (x - a_i), g) for many
/// polynomials g of one fixed length.
///
/// This follows the scaled remainder tree approach of Bostan, Lecerf and
/// Schost as implemented in the SQIsign reference implementation: g is
/// reduced once modulo the root of the tree using the precomputed
/// reciprocal, pre-scaled by the same reciprocal, and then pushed down the
/// tree where each step costs a single middle product of half the size.
/// The product of the leaf values equals Res(hI, g) up to a scalar which
/// depends only on the tree and on g.len(); the ratios of resultants taken
/// in sqrt-velu are therefore exact.
///
/// For a small number of roots the tree does not pay off and plain Horner
/// evaluation at each root is used instead.
#[derive(Clone, Debug)]
pub struct ResultantTree<Fp: FpTrait> {
    /// Number of roots.
    n: usize,
    /// The (fixed) length of the polynomials which can be evaluated.
    glen: usize,
    /// Array-encoded binary tree: the children of node i are the nodes
    /// 2i + 1 and 2i + 2. Each node holds a monic polynomial, the product
    /// of the monic linear leaves below it, with n_sub + 1 coefficients.
    nodes: Vec<Vec<Fp>>,
    /// The roots themselves, kept for the Horner fallback.
    roots: Vec<Fp>,
    /// Projective reciprocal of the reversed root polynomial, satisfying
    /// rev(root) * r0 = a0 mod x^max(n, glen - n).
    r0: Vec<Fp>,
    a0: Fp,
    /// Reusable scratch space for resultant computations.
    scratch: Vec<Fp>,
}

impl<Fp: FpTrait> ResultantTree<Fp> {
    /// Number of roots below which the scaled remainder tree loses to
    /// plain Horner evaluation and the tree is not built at all.
    ///
    /// Measured for CSIDH-512 sized fields: per resultant the tree breaks
    /// even with Horner at about 16-20 roots and is ~20% faster at 24 and
    /// ~30% faster at 32 roots (see benches/bench_poly_mul.rs), so trees
    /// only engage for very large ell (around 1600+), beyond anything in
    /// CSIDH-512 where size_I <= 12.
    const HORNER_THRESHOLD: usize = 20;

    /// Build the product tree and root reciprocal for a set of roots, for
    /// computing resultants against polynomials of length exactly glen.
    pub fn new_from_roots(roots: &[Fp], glen: usize) -> Self {
        let n = roots.len();
        debug_assert!(n > 0);

        if n < Self::HORNER_THRESHOLD {
            return Self {
                n,
                glen,
                nodes: Vec::new(),
                roots: roots.to_vec(),
                r0: Vec::new(),
                a0: Fp::ONE,
                scratch: Vec::new(),
            };
        }

        // Recursively build the product tree of the monic linear factors.
        let mut nodes: Vec<Vec<Fp>> = Vec::new();
        Self::build_node(&mut nodes, 0, roots);

        // Projective reciprocal of the reversed root polynomial: this is
        // needed to precision glen - n for the reduction of g modulo the
        // root and to precision n for the pre-scaling of the remainder.
        let lenr = n.max(glen.saturating_sub(n));
        let mut frev = vec![Fp::ZERO; (n + 1).max(lenr)];
        for i in 0..=n {
            frev[i] = nodes[0][n - i];
        }
        let mut r0 = vec![Fp::ZERO; lenr];
        let mut rec_scratch = vec![Fp::ZERO; Polynomial::<Fp>::reciprocal_scratch_len(lenr)];
        let a0 = Polynomial::reciprocal_into(&mut r0, &frev, lenr, &mut rec_scratch);

        let mut tree = Self {
            n,
            glen,
            nodes,
            roots: roots.to_vec(),
            r0,
            a0,
            scratch: Vec::new(),
        };

        // One reusable scratch buffer for all resultant computations:
        // three length-n working buffers plus the maximum of the scratch
        // requirements of the reduction, pre-scaling and tree descent.
        let lenq = glen.saturating_sub(n);
        let mut req = Polynomial::<Fp>::reduce_mod_scratch_len(n + lenq, n + 1)
            .max(Polynomial::<Fp>::mul_low_scratch_len(n));
        if n > 1 {
            let n1 = n >> 1;
            let n0 = n - n1;
            req = req
                .max(tree.descend_scratch_len(1, n0, n))
                .max(tree.descend_scratch_len(2, n1, n));
        }
        tree.scratch = vec![Fp::ZERO; 4 * n + req];
        tree
    }

    fn build_node(nodes: &mut Vec<Vec<Fp>>, idx: usize, roots: &[Fp]) {
        if idx >= nodes.len() {
            nodes.resize(idx + 1, Vec::new());
        }
        let n = roots.len();
        if n == 1 {
            nodes[idx] = vec![-roots[0], Fp::ONE];
            return;
        }
        let n1 = n >> 1;
        let n0 = n - n1;
        Self::build_node(nodes, 2 * idx + 1, &roots[..n0]);
        Self::build_node(nodes, 2 * idx + 2, &roots[n0..]);
        let mut prod = vec![Fp::ZERO; n + 1];
        Polynomial::mul_into(&mut prod, &nodes[2 * idx + 1], &nodes[2 * idx + 2]);
        nodes[idx] = prod;
    }

    /// Scratch length required by `descend` for the node idx covering
    /// nleaves roots, whose parent buffer has length parent_len.
    fn descend_scratch_len(&self, idx: usize, nleaves: usize, parent_len: usize) -> usize {
        let brother = idx - 1 + 2 * (idx & 1);
        let blen = self.nodes[brother].len();
        let mut req = Polynomial::<Fp>::mul_middle_scratch_len(blen, parent_len);
        if nleaves > 1 {
            let n1 = nleaves >> 1;
            let n0 = nleaves - n1;
            req = req
                .max(self.descend_scratch_len(2 * idx + 1, n0, blen))
                .max(self.descend_scratch_len(2 * idx + 2, n1, blen));
        }
        blen + req
    }

    /// Compute the resultant of prod_i (x - a_i) with g, scaled by a
    /// non-zero constant which depends only on the tree and g.len().
    ///
    /// For polynomials of equal length the constant is identical, so the
    /// ratios of two outputs are exact; this is the only way the value is
    /// consumed in sqrt-velu (numerator and denominator of a projective
    /// point or curve constant).
    pub fn resultant(&mut self, g: &[Fp]) -> Fp {
        // Horner fallback for small trees.
        if self.nodes.is_empty() {
            let mut res = Fp::ONE;
            for a in self.roots.iter() {
                let mut acc = g[g.len() - 1];
                for c in g.iter().rev().skip(1) {
                    acc = acc * *a + *c;
                }
                res *= acc;
            }
            return res;
        }

        debug_assert!(g.len() == self.glen);
        let n = self.n;

        let mut scratch = std::mem::take(&mut self.scratch);
        let res = {
            let (buf_a, rest) = scratch.split_at_mut(n);
            let (buf_b, rest) = rest.split_at_mut(n);
            let (rem, rest) = rest.split_at_mut(n);

            // Reduce g modulo the root polynomial (scaled by a0).
            if self.glen > n {
                Polynomial::reduce_mod_scaled_into(
                    buf_a,
                    g,
                    &self.nodes[0],
                    &self.r0[..self.glen - n],
                    &self.a0,
                    rest,
                );
            } else {
                buf_a[..g.len()].copy_from_slice(g);
                for x in buf_a[g.len()..].iter_mut() {
                    *x = Fp::ZERO;
                }
            }

            // Pre-scale: G = rev(rev(g mod root) * r0 mod x^n), after which
            // each step down the tree is a single middle product.
            for i in 0..n {
                buf_b[i] = buf_a[n - 1 - i];
            }
            Polynomial::mul_low_into(buf_a, buf_b, &self.r0[..n], n, rest);
            for i in 0..n {
                buf_b[i] = buf_a[n - 1 - i];
            }

            if n == 1 {
                rem[0] = buf_b[0];
            } else {
                let n1 = n >> 1;
                let n0 = n - n1;
                self.descend(1, n0, buf_b, &mut rem[..n0], rest);
                self.descend(2, n1, buf_b, &mut rem[n0..], rest);
            }

            let mut res = rem[0];
            for r in rem[1..].iter() {
                res *= *r;
            }
            res
        };
        self.scratch = scratch;
        res
    }

    /// Push the scaled remainder of the parent of node idx down to the
    /// leaves below idx, writing one (scaled) evaluation per leaf to rem.
    fn descend(
        &self,
        idx: usize,
        nleaves: usize,
        parent_g: &[Fp],
        rem: &mut [Fp],
        scratch: &mut [Fp],
    ) {
        let brother = idx - 1 + 2 * (idx & 1);
        let blen = self.nodes[brother].len();
        let (fg, rest) = scratch.split_at_mut(blen);
        Polynomial::mul_middle_into(fg, &self.nodes[brother], parent_g, rest);

        if nleaves == 1 {
            rem[0] = fg[blen - 1];
            return;
        }
        let n1 = nleaves >> 1;
        let n0 = nleaves - n1;
        self.descend(2 * idx + 1, n0, fg, &mut rem[..n0], rest);
        self.descend(2 * idx + 2, n1, fg, &mut rem[n0..], rest);
    }
}

impl<Fp: FpTrait> Poly<Fp> for Polynomial<Fp> {
    fn new_from_ele(a: &Fp) -> Self {
        Self::new_from_ele(a)
    }
    fn new_from_slice(a: &[Fp]) -> Self {
        Self::new_from_slice(a)
    }
    fn set_from_slice(&mut self, a: &[Fp]) {
        self.set_from_slice(a)
    }

    fn degree(&self) -> Option<usize> {
        self.degree()
    }

    fn coefficients(&self) -> &[Fp] {
        &self.coeffs
    }

    fn reverse(&self) -> Self {
        self.reverse()
    }

    fn scale(&self, a: &Fp) -> Self {
        self.scale(a)
    }

    fn evaluate(&self, a: &Fp) -> Fp {
        self.evaluate(a)
    }

    fn product_from_quadratic_leaves(leaves: &[[Fp; 3]]) -> Self {
        Self::product_from_quadratic_leaves(leaves)
    }

    fn product_from_palindromic_leaves(leaves: &[[Fp; 3]]) -> Self {
        Self::product_from_palindromic_quadratic_leaves(leaves)
    }

    fn resultant_from_roots(&self, ai: &[Fp]) -> Fp {
        self.resultant_from_roots(ai)
    }
}

impl<Fp: FpTrait> Index<usize> for Polynomial<Fp> {
    type Output = Fp;

    fn index(&self, index: usize) -> &Self::Output {
        self.coeffs.get(index).expect("Index out of bounds")
    }
}

impl<Fp: FpTrait> IndexMut<usize> for Polynomial<Fp> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.coeffs.get_mut(index).expect("Index out of bounds")
    }
}

impl<Fp: FpTrait> Default for Polynomial<Fp> {
    fn default() -> Self {
        Self {
            coeffs: vec![Fp::ONE; 1],
        }
    }
}

impl<Fp: FpTrait> Neg for &Polynomial<Fp> {
    type Output = Polynomial<Fp>;

    #[inline(always)]
    fn neg(self) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.set_neg();
        r
    }
}

impl<Fp: FpTrait> Add<Polynomial<Fp>> for Polynomial<Fp> {
    type Output = Polynomial<Fp>;

    #[inline(always)]
    fn add(self, other: Polynomial<Fp>) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.set_add(&other);
        r
    }
}

impl<Fp: FpTrait> Add<&Polynomial<Fp>> for &Polynomial<Fp> {
    type Output = Polynomial<Fp>;

    #[inline(always)]
    fn add(self, other: &Polynomial<Fp>) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.set_add(other);
        r
    }
}

impl<Fp: FpTrait> AddAssign<Polynomial<Fp>> for Polynomial<Fp> {
    #[inline(always)]
    fn add_assign(&mut self, other: Polynomial<Fp>) {
        self.set_add(&other);
    }
}

impl<Fp: FpTrait> AddAssign<&Polynomial<Fp>> for Polynomial<Fp> {
    #[inline(always)]
    fn add_assign(&mut self, other: &Polynomial<Fp>) {
        self.set_add(other);
    }
}

impl<Fp: FpTrait> Sub<Polynomial<Fp>> for Polynomial<Fp> {
    type Output = Polynomial<Fp>;

    #[inline(always)]
    fn sub(self, other: Polynomial<Fp>) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.set_sub(&other);
        r
    }
}

impl<Fp: FpTrait> Sub<&Polynomial<Fp>> for &Polynomial<Fp> {
    type Output = Polynomial<Fp>;

    #[inline(always)]
    fn sub(self, other: &Polynomial<Fp>) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.set_sub(other);
        r
    }
}

impl<Fp: FpTrait> SubAssign<&Polynomial<Fp>> for Polynomial<Fp> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &Polynomial<Fp>) {
        self.set_sub(other);
    }
}

impl<Fp: FpTrait> Mul<&Polynomial<Fp>> for &Polynomial<Fp> {
    type Output = Polynomial<Fp>;

    #[inline(always)]
    fn mul(self, other: &Polynomial<Fp>) -> Polynomial<Fp> {
        let mut r = self.clone();
        r.set_mul(other);
        r
    }
}

// TODO: ideally we don't want to do the moving here?
impl<Fp: FpTrait> MulAssign<Polynomial<Fp>> for Polynomial<Fp> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Polynomial<Fp>) {
        self.set_mul(&other);
    }
}

impl<Fp: FpTrait> MulAssign<&Polynomial<Fp>> for Polynomial<Fp> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &Polynomial<Fp>) {
        self.set_mul(other);
    }
}

impl<Fp: FpTrait> MulAssign<Fp> for Polynomial<Fp> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Fp) {
        self.scale_into(&other);
    }
}

impl<Fp: FpTrait> MulAssign<&Fp> for Polynomial<Fp> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &Fp) {
        self.scale_into(other);
    }
}

impl<Fp: FpTrait> ::std::fmt::Display for Polynomial<Fp> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if i == 0 {
                write!(f, "({c})")?
            } else {
                write!(f, "({c})*x^{i} + ")?
            }
        }
        Ok(())
    }
}
