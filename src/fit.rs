use ndarray::{Array1, ArrayView1, ArrayViewMut1};

pub fn sparse_categorical_crossentropy(y_true: usize, y_pred: &Array1<f32>) -> f32 {
    const EPSILON: f32 = 1e-7;
    -(y_pred[y_true].max(EPSILON).ln())
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    pub m: Vec<Array1<f32>>,
    pub v: Vec<Array1<f32>>,
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

    pub fn update(&mut self, params: &mut [ArrayViewMut1<f32>], grads: &[ArrayView1<f32>]) {
        //println!("Updating parameters using Adam...");
        if self.m.is_empty() {
            self.m = params.iter().map(|p| Array1::zeros(p.dim())).collect();
            self.v = params.iter().map(|p| Array1::zeros(p.dim())).collect();
        }

        self.t += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.t as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.t as i32));

        for ((param, grad), (m, v)) in params
            .iter_mut()
            .zip(grads)
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            *m = self.beta1 * &*m + (1.0 - self.beta1) * grad;
            *v = self.beta2 * &*v + (1.0 - self.beta2) * grad.mapv(|x| x * x);
            *param -= &(lr_t * &*m / (v.mapv(|x| x.sqrt()) + self.epsilon));
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
