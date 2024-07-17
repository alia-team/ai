use log::error;
use ndarray::{
    Array, Array1, ArrayBase, ArrayView1, ArrayViewMut1, Dim, IxDyn, IxDynImpl, OwnedRepr, ViewRepr,
};

pub fn sparse_categorical_crossentropy(y_true: usize, y_pred: &Array1<f32>, verbose: bool) -> f32 {
    if verbose {
        println!("Target: {}", y_true);
    }
    if y_true >= y_pred.len() {
        println!(
            "Warning: y_true ({}) is out of bounds for y_pred of length {}",
            y_true,
            y_pred.len()
        );
        return f32::MAX;
    }
    let epsilon = 1e-7;

    if verbose {
        println!("Output before softmax: {:?}", y_pred.to_vec());
    }

    // Apply softmax
    let max_val = y_pred.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = y_pred.mapv(|a| (a - max_val).exp());
    let sum = exp.sum();
    let softmax = exp / (sum + epsilon);

    if verbose {
        println!("Output after softmax: {:?}", softmax.to_vec());
    }

    // Get the predicted class (index of the highest value)
    let predicted_class = softmax
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap();

    if verbose {
        println!("Predicted class: {}", predicted_class);
    }

    -((softmax[y_true] + epsilon).ln())
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    pub m: Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>,
    pub v: Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn update(
        &mut self,
        params: &mut [&mut ArrayBase<ViewRepr<&mut f32>, IxDyn>],
        grads: &[ArrayBase<ViewRepr<&f32>, IxDyn>],
    ) {
        if params.len() != grads.len() {
            error!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params.len(),
                grads.len()
            );
            panic!("Number of parameter arrays does not match number of gradient arrays");
        }

        if self.m.is_empty() {
            self.m = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
            self.v = params.iter().map(|p| Array::zeros(p.raw_dim())).collect();
        }

        if self.m.len() != params.len() || self.v.len() != params.len() {
            error!("Mismatch in number of parameter arrays ({}) and optimizer state arrays (m: {}, v: {})", params.len(), self.m.len(), self.v.len());
            panic!("Mismatch in number of parameter arrays and optimizer state arrays");
        }

        self.t += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.t as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.t as i32));

        for (i, ((param, grad), (m, v))) in params
            .iter_mut()
            .zip(grads)
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
            .enumerate()
        {
            if param.shape() != grad.shape()
                || param.shape() != m.shape()
                || param.shape() != v.shape()
            {
                error!(
                    "Dimension mismatch for parameter {}: param {:?}, grad {:?}, m {:?}, v {:?}",
                    i,
                    param.shape(),
                    grad.shape(),
                    m.shape(),
                    v.shape()
                );
                panic!(
                    "Dimension mismatch between parameter, gradient, and optimizer state arrays"
                );
            }

            *m = &(self.beta1 * &*m) + &((1.0 - self.beta1) * grad);
            *v = &(self.beta2 * &*v) + &((1.0 - self.beta2) * &grad.mapv(|x| x * x));

            let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.t as i32)));
            let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.t as i32)));

            let update = &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + self.epsilon)) * lr_t;
            **param -= &update;
        }
    }
}

pub struct LRScheduler {
    initial_lr: f32,
    decay_rate: f32,
    decay_steps: usize,
}

impl LRScheduler {
    pub fn new(initial_lr: f32, decay_rate: f32, decay_steps: usize) -> Self {
        LRScheduler {
            initial_lr,
            decay_rate,
            decay_steps,
        }
    }

    pub fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr
            * self
                .decay_rate
                .powf((step as f32) / (self.decay_steps as f32))
    }
}
