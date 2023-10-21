use std::{fmt::Debug, ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign}};
use crate::{Matrix2D, Zero, One, Epsilon, MatrixError, MatrixLUP};

/*
 * pub fn solve_forward()
 * 
 * Descriptions:
 * L * x = B
 * input: 
 *  - L: shape (N, N) 
 *  - B: shape (N, 1) 
 * output: 
 *  - x: shape (N, 1) 
 */
#[allow(dead_code)]
pub fn solve_forward<T>(l: &Matrix2D<T>, b: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> 
    where T: Zero + One + Clone + Copy + Debug + PartialEq + PartialOrd
        + Add<Output = T> + AddAssign 
        + Sub<Output = T> + SubAssign 
        + Mul<Output = T> + MulAssign
        + Div<Output = T> + DivAssign
        + std::ops::Neg<Output = T>
        + Epsilon
{
    if !l.is_square {
        return Err(MatrixError::NotSquare);
    }
    if l.shape.0!=b.shape.0 {
        return Err(MatrixError::ShapeMismatch);
    }

    let n = l.shape.0;
    let mut x = Matrix2D::new(n, 1);
    let mut tmp;
    for i in 0..n {
        tmp = b.data[i][0];
        for j in 0..i {
            tmp -= l.data[i][j] * x.data[j][0];
        }
        x.data[i][0] = tmp / l.data[i][i];
    }
    Ok(x)
}

/*
 * pub fn solve_backward()
 * 
 * Descriptions:
 * U * x = B
 * input: 
 *  - U: shape (N, N) 
 *  - B: shape (N, 1) 
 * output: 
 *  - x: shape (N, 1) 
 */
#[allow(dead_code)]
pub fn solve_backward<T>(u: &Matrix2D<T>, b: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> 
    where T: Zero + One + Clone + Copy + Debug + PartialEq + PartialOrd
        + Add<Output = T> + AddAssign 
        + Sub<Output = T> + SubAssign 
        + Mul<Output = T> + MulAssign
        + Div<Output = T> + DivAssign
        + std::ops::Neg<Output = T>
        + Epsilon
{
    if !u.is_square {
        return Err(MatrixError::NotSquare);
    }
    if u.shape.0!=b.shape.0 {
        return Err(MatrixError::ShapeMismatch);
    }

    let n = u.shape.0;
    let mut x = Matrix2D::new(n, 1);
    let mut tmp;
    for i in (0..n).rev() {
        tmp = b.data[i][0];
        for j in i..n {
            tmp -= u.data[i][j] * x.data[j][0];
        }
        x.data[i][0] = tmp / u.data[i][i];
    }
    Ok(x)
}

/*
 * pub fn solve()
 * 
 * Descriptions:
 * A * x = B
 * input: 
 *  - A: shape (N, N) 
 *  - B: shape (N, 1) 
 * output: 
 *  - x: shape (N, 1) 
 * 
 * Calculation:
 *    A * x = B
 * => P * A * x = P * B
 * ie. P * A = L * U
 * => L * U * x = P * B
 * let U * x = y
 * => L * y = P * b
 * get y by solve_forward()
 * and use solve_backward() for U * x = y to figure out x
 */
#[allow(dead_code)]
pub fn solve<T>(lup: &MatrixLUP<T>, b: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> 
    where T: Zero + One + Clone + Copy + Debug + PartialEq + PartialOrd
        + Add<Output = T> + AddAssign 
        + Sub<Output = T> + SubAssign 
        + Mul<Output = T> + MulAssign
        + Div<Output = T> + DivAssign
        + std::ops::Neg<Output = T>
        + Epsilon
{
    if !lup.l.is_square {
        return Err(MatrixError::NotSquare);
    }
    if lup.l.shape.0!=b.shape.0 {
        return Err(MatrixError::ShapeMismatch);
    }

    let pb = lup.p.dot(b)?;
    let y = solve_forward(&lup.l, &pb)?;
    let x = solve_backward(&lup.u, &y)?;
    Ok(x)
}


