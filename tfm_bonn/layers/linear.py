import cupy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base_layer import Layer
from ..utils.activations import f_factory, sigmoid
from ..utils.thresholds import pc_factory
from ..utils.bias_init import bias_factory
from ..utils.regularizers import gradient_factory


class LinearConstant(Layer):
    """
    Capa lineal con pesos binarios fijos y sesgos aprendidos, diseñada para inducir codificación dispersa.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 pretrained_weights=None,
                 weight_learned=False,
                 pc=0.1,
                 s=0.1,
                 reg=False,
                 pc_method="constant",
                 bias_method="simple",
                 act="sigmoid",
                 reg_method="mse",
                 constant=[0.1],
                 constant_type="fix",
                 noise_type="Any",
                 noise_level=0.1):

        self.weight = pc_factory(input_dim, output_dim, pc=pc, tipo=pc_method,
                                 pretrained_weights=pretrained_weights).astype('float64').T
        self.bias = bias_factory(s, self.weight, tipo=bias_method)
        self.f, self.df = f_factory(act)

        self.outputs = 0
        self.gradient = 0
        self.dW = 0
        self.db = 0

        self.weight_learned = weight_learned
        self.first_weight = self.weight.copy()
        self.hidden_dim = output_dim

        X = self.weight.get()
        self.sim_matrx = np.array(cosine_similarity(X.T))

        self.s = s
        self.reg = reg
        self.reg_method = reg_method
        self.noise_type = noise_type
        self.noise_level = noise_level

        self.constant_type = constant_type
        self.cte = constant
        self.constant = [c if callable(c) else c for c in constant]

    def forward(self, x, training=True):
        self.inputs = x.copy()
        self.outputs = self.f(np.dot(self.inputs, self.weight) - self.bias)
        return self.outputs.copy()

    def predict(self, x):
        return sigmoid(np.dot(x, self.weight) - self.bias).get()

    def backward(self, next_layer=None, bias_learned=True):
        if not bias_learned:
            return self.gradient, 0, 0

        activation_derivative = self.df(self.outputs)
        self.gradient = np.dot(next_layer.gradient, next_layer.weight.T)

        activity_term = self.cte[0] * \
            (np.mean(self.outputs) - self.s) if self.reg else 0
        combined_gradient = activation_derivative * \
            (self.gradient + activity_term)

        self.db = np.sum(combined_gradient, axis=0)

        if self.weight_learned:
            self.dW = self.inputs.T.dot(self.gradient).astype('float64')

        return self.gradient, self.db, None

    def step(self, lr=0.01):
        self.bias += lr * self.db
        if self.weight_learned:
            self.weight -= lr * self.dW
