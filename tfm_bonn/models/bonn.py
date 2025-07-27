import cupy as np
from .base_model import Model
from ..layers.linear import LinearConstant
from ..layers.last_linear import LastLinear


class BONN(Model):
    """
    Modelo de red neuronal multicapa (MLP) bioinspirado, con una capa oculta dispersa y una capa de salida densa.
    """

    def __init__(self,
                 hidden_layer_mult=10,
                 s1=0.1,
                 pc=0.1,
                 reg=False,
                 constant1=1,
                 lr=0.01,
                 weight_learned=False,
                 pc_method="constant",
                 bias_method="simple",
                 act="sigmoid",
                 reg_method="kl",
                 pretrained_weights=None,
                 noise_type="Any",
                 noise_level=0.1,
                 mode="ALL",
                 tam_img=15,
                 output_dim=10):

        self.input_dim = 2 * tam_img * \
            tam_img if mode != "perceptron" else 2 * tam_img * tam_img
        self.hidden_dim = int(self.input_dim * hidden_layer_mult)
        self.lr = lr
        self.mode = mode

        self.first = LinearConstant(
            self.input_dim,
            self.hidden_dim,
            weight_learned=weight_learned,
            s=s1,
            pc=pc,
            reg=reg,
            pc_method=pc_method,
            bias_method=bias_method,
            act=act,
            reg_method=reg_method,
            constant=[constant1],
            pretrained_weights=pretrained_weights,
            noise_type=noise_type,
            noise_level=noise_level
        )

        self.output = LastLinear(self.hidden_dim, output_dim)
        self.first_weight = self.first.weight.copy()

    def forward(self, x):
        y = self.first.forward(x)
        out = self.output.forward(y)
        return out, y

    def regime_experiment(self, x):
        output = self.first.forward(x)
        y = self.outputs.dot(self.output.weight[:, :2])
        return y

    def predict(self, x):
        y = self.first.predict(x)
        out = self.output.predict(y)
        return out

    def backward(self, targets, weight_train=False, bias_learned=True):
        _, self.db2, self.dW2 = self.output.backward(
            targets, weight_train=weight_train)
        _, self.db, self.dW = self.first.backward(
            next_layer=self.output, bias_learned=bias_learned)

    def step(self):
        self.first.step(self.lr)
        self.output.step(self.lr)
