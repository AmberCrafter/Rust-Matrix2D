use crate::{Epsilon, Matrix2D, MatrixError, One, ScalarOperation, Zero};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

trait Float {
    fn wrap_sqrt(&self) -> Self;
}

impl Float for f64 {
    fn wrap_sqrt(&self) -> Self {
        self.sqrt()
    }
}

impl Float for f32 {
    fn wrap_sqrt(&self) -> Self {
        self.sqrt()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixQR<T> {
    q: Matrix2D<T>,
    r: Matrix2D<T>,
}

trait GramSchmidt<T>
where
    T: Zero
        + One
        + Clone
        + Copy
        + Debug
        + PartialEq
        + PartialOrd
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Mul<Output = T>
        + MulAssign
        + Div<Output = T>
        + DivAssign
        + std::ops::Neg<Output = T>
        + Epsilon
        + Float,
    Vec<T>: ScalarOperation<T>,
{
    fn qr_decomposition(&self) -> Result<MatrixQR<T>, MatrixError>;
    fn norm2(v: &Vec<T>) -> T {
        v.iter().fold(T::zero(), |acc, x| acc + *x * *x).wrap_sqrt()
    }

    fn unit_vector(v: &Vec<T>) -> Vec<T> {
        let norm = Self::norm2(v);
        v.iter().map(|&x| x / norm).collect()
    }

    fn dot_vector(v1: &Vec<T>, v2: &Vec<T>) -> T {
        v1.iter()
            .zip(v2.iter())
            .fold(T::zero(), |acc, (x1, x2)| acc + *x1 * *x2)
    }
}

impl<T> GramSchmidt<T> for Matrix2D<T>
where
    T: Zero
        + One
        + Clone
        + Copy
        + Debug
        + PartialEq
        + PartialOrd
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Mul<Output = T>
        + MulAssign
        + Div<Output = T>
        + DivAssign
        + std::ops::Neg<Output = T>
        + Epsilon
        + Float,
    Vec<T>: ScalarOperation<T>,
    Matrix2D<T>: From<Vec<Vec<T>>>,
{
    fn qr_decomposition(&self) -> Result<MatrixQR<T>, MatrixError> {
        let mut unit_vectors = Vec::new();
        for col in 0..self.shape.1 {
            unit_vectors.push(Self::unit_vector(
                &(0..self.shape.0).map(|row| self.data[row][col]).collect(),
            ));
        }

        let at = self.transpose();

        let mut ut = Vec::new();
        let mut et = Vec::new();

        for i in 0..at.shape.0 {
            let mut tmp = at.data[i].clone();
            let mut tmp2 = vec![T::zero(); at.shape.1];
            for l in 0..i {
                for j in 0..at.shape.1 {
                    let dot = <Matrix2D<T> as GramSchmidt<T>>::dot_vector(&at.data[i], &et[l]);
                    tmp2[j] += et[l][j] * dot;
                }
            }
            for j in 0..at.shape.1 {
                tmp[j] -= tmp2[j];
            }
            let norm = <Matrix2D<T> as GramSchmidt<T>>::norm2(&tmp);
            et.push(tmp.iter().map(|x| *x / norm).collect::<Vec<_>>());
            ut.push(tmp);
        }

        let mut r = Vec::new();
        for i in 0..et.len() {
            let mut tmp = Vec::new();
            for j in 0..at.shape.0 {
                tmp.push(<Matrix2D<T> as GramSchmidt<T>>::dot_vector(
                    &at.data[j],
                    &et[i],
                ));
            }
            r.push(tmp);
        }

        let r = Matrix2D::from(r);
        let q = Matrix2D::from(et).transpose();

        let res = MatrixQR { q, r };
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::modules::gram_schmidt::GramSchmidt;

    use super::Matrix2D;

    #[test]
    fn case1() {
        let v = vec![3.0, 4.0];
        let nv = Matrix2D::norm2(&v);

        assert_eq!(nv, 5.0);
    }

    #[test]
    fn case2() {
        let m1 = Matrix2D::from(vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ]);

        m1.qr_decomposition().unwrap();
    }
}
