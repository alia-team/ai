pub struct Tensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> Tensor<'a> {
        Tensor { data, shape }
    }
}
