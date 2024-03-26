use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq)]
pub struct Tensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

#[derive(Debug, PartialEq)]
pub enum TensorError {
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    IndexOutOfBounds {
        dimension: usize,
        index: usize,
        bound: usize,
    },
    ShapeDataMismatch {
        shape_elements: usize,
        data_elements: usize,
    },
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> Result<Tensor<'a>, TensorError> {
        let shape_elements = shape.iter().product::<usize>();
        if shape_elements != data.len() {
            return Err(TensorError::ShapeDataMismatch {
                shape_elements,
                data_elements: data.len(),
            });
        }
        Ok(Tensor { data, shape })
    }

    pub fn flat_index(&self, shape: &[usize]) -> Result<usize, TensorError> {
        if shape.len() != self.shape.len() {
            return Err(TensorError::DimensionMismatch {
                expected: self.shape.len(),
                found: shape.len(),
            });
        }

        let mut flat_index = 0;
        let mut dimension_stride = 1;

        for (i, &dimension_index) in shape.iter().rev().enumerate() {
            let shape_index = self.shape.len() - 1 - i;

            if dimension_index >= self.shape[shape_index] {
                return Err(TensorError::IndexOutOfBounds {
                    dimension: shape_index,
                    index: dimension_index,
                    bound: self.shape[shape_index] - 1,
                });
            }
            flat_index += dimension_index * dimension_stride;
            dimension_stride *= self.shape[shape_index];
        }

        Ok(flat_index)
    }

    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let flat_index = self.flat_index(index)?;
        Ok(self.data[flat_index])
    }
}

impl Display for TensorError {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match *self {
            TensorError::DimensionMismatch { expected, found } => {
                write!(
                    formatter,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            TensorError::IndexOutOfBounds {
                dimension,
                index,
                bound,
            } => {
                write!(
                    formatter,
                    "Index out of bounds: dimension {}, index {}, exceeds bound {}",
                    dimension, index, bound
                )
            }
            TensorError::ShapeDataMismatch {
                shape_elements,
                data_elements,
            } => {
                write!(
                    formatter,
                    "Shape-data mismatch: expected shape to represent {} elements, but data contains {} elements.",
                    shape_elements,
                    data_elements
                )
            }
        }
    }
}
