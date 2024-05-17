use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
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
    ShapeMismatch {
        a: Vec<usize>,
        b: Vec<usize>,
    },
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, TensorError> {
        let shape_elements = shape.iter().product::<usize>();
        if shape_elements != data.len() {
            return Err(TensorError::ShapeDataMismatch {
                shape_elements,
                data_elements: data.len(),
            });
        }
        Ok(Tensor { data, shape })
    }

    pub fn flat_index(&self, index: Vec<usize>) -> Result<usize, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::DimensionMismatch {
                expected: self.shape.len(),
                found: index.len(),
            });
        }

        let mut flat_index = 0;
        let mut dimension_stride = 1; // Tells us how many elements to skip for
                                      // moving to the next dimension

        // Iterate in reverse to start from the innermost dimension, which
        // simplify dimension stride calculation
        for (i, &dimension_index) in index.iter().rev().enumerate() {
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

    pub fn get(&self, index: Vec<usize>) -> Result<f64, TensorError> {
        let flat_index = self.flat_index(index)?;
        Ok(self.data[flat_index])
    }

    pub fn dot(&self, tensor: &Tensor) -> Result<f64, TensorError> {
        if self.shape != tensor.shape {
            return Err(TensorError::ShapeMismatch {
                a: self.shape.clone(),
                b: tensor.shape.clone(),
            });
        }

        Ok(self
            .data
            .iter()
            .zip(tensor.data.iter())
            .map(|(&x, &y)| x * y)
            .sum())
    }
}

impl Display for TensorError {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match self {
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
            TensorError::ShapeMismatch { a, b } => {
                write!(
                    formatter,
                    "Shape mismatch: first tensor has shape {:?} but second one has {:?}.",
                    a, b
                )
            }
        }
    }
}
