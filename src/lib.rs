use std::{fmt::{Debug, Display}, error::Error, ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign}};
mod modules;


#[allow(dead_code)]
type BoxError = Box<dyn Error + 'static>;

#[derive(Debug)]
pub enum MatrixError {
    Unknown = 0,
    NotSquare,
    OutOfBound,
    ShapeMismatch,
    SingularMatrix,
}

impl Error for MatrixError {
    fn cause(&self) -> Option<&dyn Error> {
        None
    }
    fn description(&self) -> &str {
        match self {
            MatrixError::Unknown => "Unknown Error",
            MatrixError::NotSquare => "Not Square Matrix!",
            MatrixError::OutOfBound => "Out Of Boundary",
            MatrixError::ShapeMismatch => "Shape Mismatch",
            MatrixError::SingularMatrix => "Singular Matrix",
        }
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            MatrixError::Unknown => "Unknown",
            MatrixError::NotSquare => "NotSquare",
            MatrixError::OutOfBound => "OutOfBound",
            MatrixError::ShapeMismatch => "ShapeMismatch",
            MatrixError::SingularMatrix => "SingularMatrix",
        };
        write!(f, "{}", text)
    }
}

pub trait Number {}

pub trait Zero: Sized {
    fn zero() -> Self;
}
pub trait One: Sized {
    fn one() -> Self;
}

pub trait Epsilon: Sized {
    fn epsilon() -> Self;
}

macro_rules! impl_number {
    ($dtype: ty) => {
        impl Number for $dtype {}
    };
}

macro_rules! impl_zero {
    ($dtype: ty) => {
        impl Zero for $dtype {
            fn zero() -> Self {
                0 as $dtype
            }
        }
    };
}

macro_rules! impl_one {
    ($dtype: ty, int) => {
        impl One for $dtype {
            fn one() -> Self {
                1
            }
        }
    };
    ($dtype: ty, float) => {
        impl One for $dtype {
            fn one() -> Self {
                1.0
            }
        }
    };
}

macro_rules! impl_epsilon {
    ($dtype: ty, int) => {
        impl Epsilon for $dtype {
            fn epsilon() -> Self {
                Self::zero()
            }
        }
    };
    ($dtype: ty, float) => {
        impl Epsilon for $dtype {
            fn epsilon() -> Self {
                Self::EPSILON
            }
        }
    };
}

macro_rules! impl_macro_for_all_nums {
    ($macro: tt) => {
        $macro!(i8);
        $macro!(i16);
        $macro!(i32);
        $macro!(i64);
        $macro!(i128);
        $macro!(isize);
        $macro!(u8);
        $macro!(u16);
        $macro!(u32);
        $macro!(u64);
        $macro!(u128);
        $macro!(usize);
        $macro!(f32);
        $macro!(f64);
    };

    ($macro: tt, int) => {
        $macro!(i8,     int);
        $macro!(i16,    int);
        $macro!(i32,    int);
        $macro!(i64,    int);
        $macro!(i128,   int);
        $macro!(isize,  int);
        $macro!(u8,     int);
        $macro!(u16,    int);
        $macro!(u32,    int);
        $macro!(u64,    int);
        $macro!(u128,   int);
        $macro!(usize,  int);
    };

    ($macro: tt, float) => {
        $macro!(f32,    float);
        $macro!(f64,    float);
    };
}

impl_macro_for_all_nums!(impl_number);
impl_macro_for_all_nums!(impl_zero);
impl_macro_for_all_nums!(impl_one, int);
impl_macro_for_all_nums!(impl_one, float);
impl_macro_for_all_nums!(impl_epsilon, int);
impl_macro_for_all_nums!(impl_epsilon, float);

pub trait ScalarOperation<T> {
    fn sadd(&mut self, val:T);
    fn ssub(&mut self, val:T);
    fn smul(&mut self, val:T);
    fn sdiv(&mut self, val:T);
    fn sadd_clone(&self, val:T) -> Self;
    fn ssub_clone(&self, val:T) -> Self;
    fn smul_clone(&self, val:T) -> Self;
    fn sdiv_clone(&self, val:T) -> Self;
}

impl<T> ScalarOperation<T> for Vec<T> 
    where T: Clone + Copy
    + Add<Output = T> + AddAssign 
    + Sub<Output = T> + SubAssign 
    + Mul<Output = T> + MulAssign
    + Div<Output = T> + DivAssign
{
    fn sadd(&mut self, val:T) {
        self.into_iter().for_each(|v| *v += val);
    }
    fn sdiv(&mut self, val:T) {
        self.into_iter().for_each(|v| *v -= val);
    }
    fn smul(&mut self, val:T) {
        self.into_iter().for_each(|v| *v *= val);
    }
    fn ssub(&mut self, val:T) {
        self.into_iter().for_each(|v| *v /= val);
    }
    
    fn sadd_clone(&self, val:T) -> Self {
        self.iter().map(|&v| v + val).collect()
    }
    fn sdiv_clone(&self, val:T) -> Self {
        self.iter().map(|&v| v - val).collect()
    }
    fn smul_clone(&self, val:T) -> Self {
        self.iter().map(|&v| v * val).collect()
    }
    fn ssub_clone(&self, val:T) -> Self {
        self.iter().map(|&v| v / val).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix2D<T> {
    shape: (usize, usize),
    data: Vec<Vec<T>>,
    is_square: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixLUP<T> {
    l: Matrix2D<T>,
    u: Matrix2D<T>,
    p: Matrix2D<T>,
    num_permutations: usize,
}

impl<T> ScalarOperation<T> for Matrix2D<T> 
    where T: Zero + One + Clone + Copy + Debug + PartialEq + PartialOrd
        + Add<Output = T> + AddAssign 
        + Sub<Output = T> + SubAssign 
        + Mul<Output = T> + MulAssign
        + Div<Output = T> + DivAssign
{
    fn sadd(&mut self, val: T) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] += val;
            }
        }
    }

    fn sadd_clone(&self, val:T) -> Self {
        let mut res = self.clone();
        res.sadd(val);
        res
    }

    fn ssub(&mut self, val: T) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] -= val;
            }
        }
    }

    fn ssub_clone(&self, val:T) -> Self {
        let mut res = self.clone();
        res.ssub(val);
        res
    }

    fn smul(&mut self, val: T) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] *= val;
            }
        }
    }

    fn smul_clone(&self, val:T) -> Self {
        let mut res = self.clone();
        res.smul(val);
        res
    }

    fn sdiv(&mut self, val: T) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] /= val;
            }
        }
    }

    fn sdiv_clone(&self, val:T) -> Self {
        let mut res = self.clone();
        res.sdiv(val);
        res
    }
}

#[allow(dead_code)]
impl<T> Matrix2D<T>
    where T: Zero + One + Clone + Copy + Debug + PartialEq + PartialOrd
            + Add<Output = T> + AddAssign 
            + Sub<Output = T> + SubAssign 
            + Mul<Output = T> + MulAssign
            + Div<Output = T> + DivAssign
            + std::ops::Neg<Output = T>
            + Epsilon,
          Vec<T>: ScalarOperation<T>
{
    pub fn new(row: usize, col: usize) -> Self {
        Self { shape: (row, col), data: vec![vec![T::zero(); col]; row], is_square: row==col }
    }

    pub fn transpose(&self) -> Self {
        let mut res = Self::new(self.shape.1, self.shape.0);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }

    pub fn eye(size: usize) -> Self {
        let mut m = Self::new(size, size);
        for i in 0..size {
            m.data[i][i] = T::one();
        }
        m
    }

    pub fn show(&self) {
        for val in self.data.iter() {
            println!("{:?}", val);
        }
    }

    pub fn get_col(&self, index: usize) -> Result<Matrix2D<T>, MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        let mut m = Matrix2D::new(self.shape.0, 1);
        for i in 0..self.shape.0 {
            m.data[i][0] = self.data[i][index];
        }
        Ok(m)
    }

    pub fn get_row(&self, index: usize) -> Result<Matrix2D<T>, MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        let mut m = Matrix2D::new(1, self.shape.1);
        for i in 0..self.shape.1 {
            m.data[0][i] = self.data[index][i];
        }
        Ok(m)
    }


    // single value
    pub fn set_all(&mut self, val: T) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] = val;
            }
        }
    }
    pub fn set_diag(&mut self, val: T) -> Result<(), MatrixError> {
        if !self.is_square { return Err(MatrixError::NotSquare)}
        for i in 0..self.shape.0 {
            self.data[i][i] = val;
        }
        Ok(())
    }


    // array operation
    pub fn add_row(&mut self, index: usize, arr: &Vec<T>) -> Result<(), MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        if arr.len() != self.shape.1 { return Err(MatrixError::ShapeMismatch)}
        for i in 0..self.shape.1 {
            self.data[index][i] += arr[i];
        }
        Ok(())
    }

    pub fn add_row_clone(&self, index: usize, arr: &Vec<T>) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.add_row(index, arr)?;
        Ok(m)
    }

    pub fn add_col(&mut self, index: usize, arr: &Vec<T>) -> Result<(), MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        if arr.len() != self.shape.0 { return Err(MatrixError::ShapeMismatch)}
        for i in 0..self.shape.0 {
            self.data[i][index] += arr[i];
        }
        Ok(())
    }

    pub fn add_col_clone(&self, index: usize, arr: &Vec<T>) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.add_col(index, arr)?;
        Ok(m)
    }

    pub fn sub_row(&mut self, index: usize, arr: &Vec<T>) -> Result<(), MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        if arr.len() != self.shape.1 { return Err(MatrixError::ShapeMismatch)}
        for i in 0..self.shape.1 {
            self.data[index][i] -= arr[i];
        }
        Ok(())
    }

    pub fn sub_row_clone(&self, index: usize, arr: &Vec<T>) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.sub_row(index, arr)?;
        Ok(m)
    }

    pub fn sub_col(&mut self, index: usize, arr: &Vec<T>) -> Result<(), MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        if arr.len() != self.shape.0 { return Err(MatrixError::ShapeMismatch)}
        for i in 0..self.shape.0 {
            self.data[i][index] -= arr[i];
        }
        Ok(())
    }

    pub fn sub_col_clone(&self, index: usize, arr: &Vec<T>) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.sub_col(index, arr)?;
        Ok(m)
    }

    // scalar operation
    pub fn sadd_row(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.1 {
            self.data[index][i] += val;
        }
        Ok(())
    }

    pub fn sadd_row_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.sadd_row(index, val)?;
        Ok(m)
    }

    pub fn sadd_col(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.0 {
            self.data[i][index] += val;
        }
        Ok(())
    }

    pub fn sadd_col_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.sadd_col(index, val)?;
        Ok(m)
    }

    pub fn ssub_row(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.1 {
            self.data[index][i] -= val;
        }
        Ok(())
    }

    pub fn ssub_row_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.ssub_row(index, val)?;
        Ok(m)
    }

    pub fn ssub_col(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.0 {
            self.data[i][index] -= val;
        }
        Ok(())
    }

    pub fn ssub_col_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.ssub_col(index, val)?;
        Ok(m)
    }

    pub fn smul_row(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.1 {
            self.data[index][i] *= val;
        }
        Ok(())
    }

    pub fn smul_row_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.smul_row(index, val)?;
        Ok(m)
    }

    pub fn smul_col(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.0 {
            self.data[i][index] *= val;
        }
        Ok(())
    }

    pub fn smul_col_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.smul_col(index, val)?;
        Ok(m)
    }

    pub fn sdiv_row(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.1 {
            self.data[index][i] /= val;
        }
        Ok(())
    }

    pub fn sdiv_row_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.sdiv_row(index, val)?;
        Ok(m)
    }

    pub fn sdiv_col(&mut self, index: usize, val: T) -> Result<(), MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        for i in 0..self.shape.0 {
            self.data[i][index] /= val;
        }
        Ok(())
    }

    pub fn sdiv_col_clone(&self, index: usize, val: T) -> Result<Matrix2D<T>, MatrixError> {
        let mut m = self.clone();
        m.sdiv_col(index, val)?;
        Ok(m)
    }

    pub fn remove(&self, index: usize) -> Result<Matrix2D<T>, MatrixError> {
        if index >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        let mut m = Self::new(self.shape.0 - 1, self.shape.1);
        let mut k = 0;
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                m.data[k][j] = self.data[i][j];
                k+=1;
            } 
        }
        Ok(m)
    }

    pub fn remove_col_clone(&self, index: usize) -> Result<Matrix2D<T>, MatrixError> {
        if index >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        let mut m = Self::new(self.shape.0, self.shape.1 - 1);
        let mut k = 0;
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                m.data[i][k] = self.data[i][j];
                k+=1;
            } 
        }
        Ok(m)
    }

    pub fn swap_row(&mut self, index1: usize, index2: usize) -> Result<(), MatrixError> {
        if index1 >= self.shape.0 || index2 >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        let tmp = self.data[index2].clone();
        self.data[index2] = self.data[index1].clone();
        self.data[index1] = tmp;
        Ok(())
    }

    pub fn swap_row_clone(&self, index1: usize, index2: usize) -> Result<Matrix2D<T>, MatrixError> {
        if index1 >= self.shape.0 || index2 >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        let mut m = self.clone();
        match m.swap_row(index1, index2) {
            Ok(()) => Ok(m),
            Err(e) => Err(e)
        }
    }

    pub fn swap_col(&mut self, index1: usize, index2: usize) -> Result<(), MatrixError> {
        if index1 >= self.shape.1 || index2 >= self.shape.1 { return Err(MatrixError::OutOfBound)}
        let mut tmp;
        for i in 0..self.shape.0 {
            tmp = self.data[i][index2];
            self.data[i][index2] = self.data[i][index1];
            self.data[i][index1] = tmp;
        }
        Ok(())
    }

    pub fn swap_col_clone(&self, index1: usize, index2: usize) -> Result<Matrix2D<T>, MatrixError> {
        if index1 >= self.shape.0 || index2 >= self.shape.0 { return Err(MatrixError::OutOfBound)}
        let mut m = self.clone();
        match m.swap_col(index1, index2) {
            Ok(()) => Ok(m),
            Err(e) => Err(e)
        }
    }

    pub fn vstack(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> {
        if self.shape.1!=other.shape.1 {return Err(MatrixError::ShapeMismatch);}
        let mut m = Matrix2D::new(self.shape.0 + other.shape.0, self.shape.1);
        let mut k = 0;
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                m.data[k][j] = self.data[i][j];
            }
            k+=1;
        }
        for i in 0..other.shape.0 {
            for j in 0..other.shape.1 {
                m.data[k][j] = other.data[i][j];
            }
            k+=1;
        }
        Ok(m)
    }

    pub fn hstack(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> {
        if self.shape.0!=other.shape.0 {return Err(MatrixError::ShapeMismatch);}
        let mut m = Matrix2D::new(self.shape.0, self.shape.1 + other.shape.1);
        let mut k;
        for i in 0..self.shape.0 {
            k = 0;
            for j in 0..self.shape.1 {
                m.data[i][k] = self.data[i][j];
                k+=1;
            }
            for j in 0..other.shape.1 {
                m.data[i][k] = other.data[i][j];
                k+=1;
            }
        }
        Ok(m)
    }

    pub fn add(&mut self, other: &Matrix2D<T>) -> Result<(), MatrixError> {
        if self.shape.0 != other.shape.0 || self.shape.1 != other.shape.1 {
            return Err(MatrixError::ShapeMismatch);
        }
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] += other.data[i][j];
            }
        }
        Ok(())
    }

    pub fn add_clone(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> {
        if self.shape.0 != other.shape.0 || self.shape.1 != other.shape.1 {
            return Err(MatrixError::ShapeMismatch);
        }
        let mut m = self.clone();
        m.add(other)?;
        Ok(m)
    }

    pub fn sub(&mut self, other: &Matrix2D<T>) -> Result<(), MatrixError> {
        if self.shape.0 != other.shape.0 || self.shape.1 != other.shape.1 {
            return Err(MatrixError::ShapeMismatch);
        }
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] -= other.data[i][j];
            }
        }
        Ok(())
    }

    pub fn sub_clone(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> {
        if self.shape.0 != other.shape.0 || self.shape.1 != other.shape.1 {
            return Err(MatrixError::ShapeMismatch);
        }
        let mut m = self.clone();
        m.sub(other)?;
        Ok(m)
    }

    pub fn mul(&mut self, other: &Matrix2D<T>) -> Result<(), MatrixError> {
        if self.shape.0 != other.shape.0 || self.shape.1 != other.shape.1 {
            return Err(MatrixError::ShapeMismatch);
        }
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                self.data[i][j] *= other.data[i][j];
            }
        }
        Ok(())
    }

    pub fn mul_clone(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> {
        if self.shape.0 != other.shape.0 || self.shape.1 != other.shape.1 {
            return Err(MatrixError::ShapeMismatch);
        }
        let mut m = self.clone();
        m.mul(other)?;
        Ok(m)
    }

    pub fn dot(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixError> {
        if self.shape.1 != other.shape.0 {
            return Err(MatrixError::ShapeMismatch);
        }

        let mut m = Matrix2D::new(self.shape.0, other.shape.1);
        for i in 0..self.shape.0 {
            for j in 0..other.shape.1 {
                for k in 0..self.shape.1 {
                    m.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Ok(m)
    }


    fn privotidx(&self, row: usize, col: usize) -> Option<usize> {
        let min_coff = T::epsilon();
        for i in row..self.shape.0 {
            if self.data[i][col] - min_coff > T::zero() || self.data[i][col] + min_coff < T::zero() {
                return Some(i);
            }
        }
        None
    }

    fn privotmaxidx(&self, row: usize, col: usize) -> Option<usize> {
        let min_coff = T::epsilon();
        let mut maxidx = row;
        let mut maxval: T = if self.data[row][col] >= T::zero() {
            self.data[row][col]
        } else {
            -self.data[row][col]
        };

        for i in row..self.shape.0 {
            let micol = if self.data[i][col] >= T::zero() {
                self.data[i][col]
            } else {
                -self.data[i][col]
            };
            if micol > maxval {
                maxval = micol;
                maxidx = i;
            }
        }
        
        if maxval < min_coff {
            None
        } else {
            Some(maxidx)
        }
    }

    pub fn row_echelon(&self) -> Result<Matrix2D<T>, MatrixError> {
        let min_coff = T::epsilon();
        let mut m = self.clone();
        let (mut i, mut j) = (0, 0);
        let mut pivot;

        // println!("{:?}", m.data);
        
        while j < m.shape.1 && i < m.shape.0 {
            pivot = match m.privotidx(i, j) {
                None => {
                    j+=1;
                    continue;
                },
                Some(val) => val
            };

            if pivot!=i {
                m.swap_row(i, pivot).unwrap();
            }
            
            // println!("swap: {:?}", m.data);
            m.smul_row(i, T::one() / m.data[i][j]).unwrap();
            
            // println!("smul: {:?}", m.data);
            for k in i+1..m.shape.0 {
                // println!("{:?}", k);
                if m.data[k][j] - min_coff > T::zero() || m.data[k][j] + min_coff < T::zero() {
                    let mut arr = m.data[i].clone();
                    arr = arr.smul_clone(-(m.data[k][j]));
                    // println!("arr: {:?}", arr);
                    m.add_row(k, &arr)?;
                }
            }
            i+=1;
            j+=1;
            // println!("{:?}", m.data);
        }
        Ok(m)
    }  


    pub fn reduced_row_echelon(&self) -> Result<Matrix2D<T>, MatrixError> {
        let min_coff = T::epsilon();
        let mut m = self.clone();
        let (mut i, mut j) = (0, 0);
        let mut pivot;

        // println!("{:?}", m.data);
        
        while j < m.shape.1 && i < m.shape.0 {
            pivot = match m.privotmaxidx(i, j) {
                None => {
                    j+=1;
                    continue;
                },
                Some(val) => val
            };

            if pivot!=i {
                m.swap_row(i, pivot).unwrap();
            }
            
            // println!("swap: {:?}", m.data);
            m.smul_row(i, T::one() / m.data[i][j]).unwrap();
            
            // println!("smul: {:?}", m.data);
            for k in 0..m.shape.0 {
                if i==k {continue;}
                // println!("{:?}", k);
                if m.data[k][j] - min_coff > T::zero() || m.data[k][j] + min_coff < T::zero() {
                    let mut arr = m.data[i].clone();
                    arr = arr.smul_clone(-(m.data[k][j]));
                    // println!("arr: {:?}", arr);
                    m.add_row(k, &arr)?;
                }
            }
            i+=1;
            j+=1;
            // println!("{:?}", m.data);
        }
        Ok(m)
    }
    

    // return (L, U, P, num_permutation)
    /* 
     * step:
     *  1. check square matrix
     *  2. init PA = LU
     *   - P = I
     *   - A = self matrix
     *   - L = O
     *   - U = self matrix
     *   - num_permutation = 0
     *  3. solve
     */ 
    pub fn lu_decomposition(&self) -> Result<MatrixLUP<T>, MatrixError> {
        if !self.is_square {
            return Err(MatrixError::NotSquare);
        }

        let size = self.shape.0;
        let min_coff = T::epsilon();
        let mut permutation = 0;
        let mut p = Matrix2D::eye(size);
        let mut l = Matrix2D::new(size, size);
        let mut u = self.clone();

        let mut pivot;
        let mut mult;

        for j in 0..size {
            pivot = u.privotmaxidx(j, j).unwrap();
            if !(u.data[pivot][j] - min_coff > T::zero() || u.data[pivot][j] + min_coff < T::zero()) {
                return Err(MatrixError::OutOfBound);
            }
            
            // println!("pivot: {:?}", pivot);
            if pivot!=j {
                u.swap_row(j, pivot)?;
                l.swap_row(j, pivot)?;
                p.swap_row(j, pivot)?;
                permutation += 1;
            }

            // println!("U: {:?}", U);

            for i in j+1..size {
                mult = u.data[i][j] / u.data[j][j];
                let arr = u.data[j].smul_clone(-mult);
                u.add_row(i, &arr)?;
                l.data[i][j] = mult;
            }
        }
        for i in 0..size {
            l.data[i][i] = T::one();
        }

        let lup = MatrixLUP {
            l, 
            u, 
            p, 
            num_permutations: permutation
        };

        Ok(lup)
    }

    fn get_cofactor(&self, buf: &mut Self, p: usize, q: usize, n: usize) -> Result<(), MatrixError> {
        if !self.is_square {
            return Err(MatrixError::NotSquare);
        }
        let (mut i, mut j) = (0, 0);
        for row in 0..n {
            for col in 0..n {
                if row!=p && col!=q {
                    buf.data[i][j] = self.data[row][col];
                    j+=1;
                    
                    if j==n-1 {
                        j = 0;
                        i+=1;
                    }
                }
            }
        }
        Ok(())
    }

    fn determinant(&self, n: usize) -> Result<T, MatrixError> {
        if !self.is_square {
            return Err(MatrixError::NotSquare);
        }
        let length = self.shape.0;
        if n == 1 {return Ok(self.data[0][0]);}
        let mut res = T::zero();
        let mut sign = T::one();
        let mut buf = Matrix2D::new(length, length);
        for i in 0..length {
            self.get_cofactor(&mut buf, 0, i, n)?;
            res += sign * self.data[0][i] * buf.determinant(n - 1).unwrap();
            sign = -sign;
        }
        Ok(res)
    }

    fn ajoint(&self) -> Result<Self, MatrixError> {
        if !self.is_square {
            return Err(MatrixError::NotSquare);
        }
        let length = self.shape.0;
        if length==1 {return Ok((*self).clone());}

        let mut res = Matrix2D::new(length, length);
        let mut sign;
        let mut buf = Matrix2D::new(length, length);
        for i in 0..length {
            for j in 0..length {
                self.get_cofactor(&mut buf, i, j, length)?;
                sign = if (i + j) % 2 == 0 {T::one()} else {-T::one()};
                res.data[j][i] = sign * buf.determinant(length - 1).unwrap();
            }
        }

        Ok(res)
    }

    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if !self.is_square {
            return Err(MatrixError::NotSquare);
        }
        let length = self.shape.0;
        let det = self.determinant(length).unwrap();
        if det == T::zero() {return Err(MatrixError::SingularMatrix);}

        let adj = self.ajoint().unwrap();
        let mut res = Matrix2D::new(length, length);
        for i in 0..length {
            for j in 0..length {
                res.data[i][j] = adj.data[i][j] / det;
            }
        }
        Ok(res)
    }

    // QR
    // unit test
}

impl<T> From<Vec<T>> for Matrix2D<T> 
    where T: Number
{
    fn from(value: Vec<T>) -> Self {
        let is_square = value.len()==1;
        Matrix2D { shape: (1, value.len()), data: vec![value], is_square}
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix2D<T> 
    where T: Number
{
    fn from(value: Vec<Vec<T>>) -> Self {
        let is_square = value.len()==value[0].len();
        Matrix2D { shape: (value.len(), value[0].len()), data: value, is_square}
    }
}


#[cfg(test)]
mod tests {
    use crate::Matrix2D;
    use crate::Epsilon;

    #[test]
    fn case_square_zero() {
        let m = Matrix2D::<isize>::new(3, 3);

        assert_eq!(vec![vec![0; 3]; 3], m.data);
        assert_eq!((3,3), m.shape);
        assert!(m.is_square);
    }

    #[test]
    fn case_mul_row_col() {
        let mut m = Matrix2D::<f64>::new(3, 3);
        m.set_all(1.0);
        assert_eq!(vec![vec![1.0; 3]; 3], m.data);

        // m.mul_row(1, 5.0).unwrap();
        assert_eq!(vec![
            vec![1.0; 3],
            vec![5.0; 3],
            vec![1.0; 3],
        ], m.data);
    }

    #[test]
    fn case_add_row_col() {
        let mut m = Matrix2D::<f64>::new(3, 3);
        m.set_all(1.0);
        assert_eq!(vec![vec![1.0; 3]; 3], m.data);

        m.add_row(1, &vec![5.0; 3]).unwrap();
        assert_eq!(vec![
            vec![1.0; 3],
            vec![6.0; 3],
            vec![1.0; 3],
        ], m.data);
    }

    #[test]
    fn case_div_row_col() {
        let mut m = Matrix2D::<i32>::new(3, 3);
        m.set_all(11);
        assert_eq!(vec![vec![11; 3]; 3], m.data);

        // m.div_row(1, 5).unwrap();
        assert_eq!(vec![
            vec![11; 3],
            vec![2; 3],
            vec![11; 3],
        ], m.data);
    }

    #[test]
    fn case_epsilon(){
        println!("{:?}", f64::EPSILON);
        println!("{:?}", f64::epsilon());
        println!("{:?}", f32::epsilon());
        println!("{:?}", usize::epsilon());
        println!("{:?}", isize::epsilon());
        println!("{:?}", i128::epsilon());
    }

    #[test]
    fn case_ref() {
        let m = Matrix2D::from(vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 1.0],
            vec![2.0, 7.0, 8.0],
        ]);
        let mref = m.row_echelon().unwrap();
        println!("{:#?}", mref);
    }

    #[test]
    fn case_rref() {
        let m = Matrix2D::from(vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 1.0],
            vec![2.0, 7.0, 8.0],
        ]);
        let mref = m.reduced_row_echelon().unwrap();
        println!("{:#?}", mref);
    }

    #[test]
    fn case_rref_1() {
        let m = Matrix2D::from(vec![
            vec![0.0, 1.0, 2.0, 4.0, 5.5],
            vec![1.0, 2.0, 1.0, 3.5, 3.0],
            vec![2.0, 7.0, 8.0, 9.0, 5.5],
            vec![2.0, 0.0, 4.0, 6.0, 5.5],
            // vec![1.0, 2.0, 4.0, 9.0, 5.5],
        ]);
        let mref = m.reduced_row_echelon().unwrap();
        println!("{:#?}", mref);
    }

    #[test]
    fn case_lu() {
        let m = Matrix2D::from(vec![
            vec![2.0, 1.0, 5.0],
            vec![4.0, 4.0, -4.0],
            vec![1.0, 3.0, 1.0],
        ]);

        let lup = m.lu_decomposition().unwrap();
        println!("{:#?}", lup.l);
        println!("{:#?}", lup.u);
        println!("{:#?}", lup.p);
        println!("{:#?}", lup.num_permutations);
    }
}