"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np


def relu(x):
    """
    ReLU is used for hidden layers so piecewise-linear subnetworks can implement
    exact comparisons and affine pieces of the symbolic update rule.
    """
    return np.maximum(0.0, x)


def relu_derivative(x):
    """
    The derivative is one on the active half-space and zero elsewhere. The result
    keeps the same dtype as the input so backward passes stay numerically aligned
    with the parameter arrays.
    """
    return (x > 0.0).astype(x.dtype)


def softmax(x):
    """
    Softmax converts a batch of logits into normalized class probabilities. The
    subtraction of the rowwise maximum prevents overflow without changing the
    resulting distribution.
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def mse_loss(predictions, targets):
    """
    Mean squared error is used for regression-style outputs such as module values
    or directly predicted scalars.
    """
    return np.mean((predictions - targets) ** 2)


def cross_entropy_from_logits(logits, targets_one_hot, eps=1e-8):
    """
    Compute cross-entropy from logits rather than from already-softmaxed values so
    training code can keep a numerically stable and conventional interface.
    """
    probs = softmax(logits)
    log_probs = np.log(probs + eps)
    loss = -np.mean(np.sum(targets_one_hot * log_probs, axis=1))
    return loss


class MLP:
    """
    MLP is a minimal dense network implementation with forward evaluation, 
    explicit backpropagation, gradient updates, and serialization. It also 
    doubles as a container for hand-crafted weights when a network is meant 
    to encode a symbolic rule exactly.
    """

    def __init__(self, layer_sizes, hidden_activation="relu", output_activation="linear", weight_scale=0.1, seed=None):
        """
        Create the parameter tensors for each affine layer. Hidden and output
        activations are configurable so the same class can serve both learned and
        hand-specified subnetworks used by the controller and modules.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output size")

        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = np.random.randn(in_dim, out_dim) * weight_scale
            b = np.zeros(out_dim, dtype=np.float32)
            self.weights.append(w.astype(np.float32))
            self.biases.append(b)


    def _hidden_forward(self, x):
        """
        Apply the configured hidden nonlinearity. The helper isolates activation
        choice from the main forward pass so the rest of the class stays compact.
        """
        if self.hidden_activation == "relu":
            return relu(x)
        elif self.hidden_activation == "tanh":
            return np.tanh(x)
        else:
            raise ValueError("Unsupported hidden activation: %s" % self.hidden_activation)


    def _hidden_backward_derivative(self, preactivations):
        """
        Return the derivative of the hidden activation evaluated at preactivations.
        Backpropagation uses preactivation values because the derivative formula is
        naturally expressed in that space for both ReLU and tanh.
        """
        if self.hidden_activation == "relu":
            return relu_derivative(preactivations)
        elif self.hidden_activation == "tanh":
            return 1.0 - np.tanh(preactivations) ** 2
        else:
            raise ValueError("Unsupported hidden activation: %s" % self.hidden_activation)


    def forward(self, x, return_cache=False):
        """
        Perform a forward pass through all affine layers. When requested, the method
        also returns the activations and preactivations needed for an explicit manual
        backward pass, which keeps the implementation transparent and framework-free.
        """
        a = x
        activations = [a]
        preactivations = []

        num_layers = len(self.weights)
        for layer_index in range(num_layers):
            w = self.weights[layer_index]
            b = self.biases[layer_index]
            z = np.matmul(a, w) + b
            preactivations.append(z)

            is_last = (layer_index == num_layers - 1)
            if is_last:
                if self.output_activation == "linear":
                    a = z
                elif self.output_activation == "softmax":
                    a = softmax(z)
                else:
                    raise ValueError("Unsupported output activation: %s" % self.output_activation)
            else:
                a = self._hidden_forward(z)
            activations.append(a)

        if not return_cache:
            return activations[-1]

        cache = {"activations": activations, "preactivations": preactivations}
        return activations[-1], cache


    def _loss_and_output_grad(self, predictions, targets, loss_type, logits_for_ce=None):
        """
        Compute the scalar loss and the gradient at the network output. This method
        centralizes the loss-specific formulas so the main backward loop can stay
        independent of whether the task is regression or classification.
        """
        batch_size = predictions.shape[0]

        if loss_type == "mse":
            loss = mse_loss(predictions, targets)
            grad_output = 2.0 * (predictions - targets) / float(batch_size)
            return loss, grad_output

        if loss_type == "cross_entropy":
            if logits_for_ce is None:
                raise ValueError("logits_for_ce must be provided for cross_entropy loss")
            loss = cross_entropy_from_logits(logits_for_ce, targets)
            probs = softmax(logits_for_ce)
            grad_output = (probs - targets) / float(batch_size)
            return loss, grad_output

        raise ValueError("Unsupported loss type: %s" % loss_type)


    def loss_and_gradients(self, x, targets, loss_type="mse"):
        """
        Run a full training-style pass: forward evaluation, loss computation, and
        reverse-mode differentiation for every parameter tensor. The returned lists
        match self.weights and self.biases elementwise so optimization code can stay
        simple and explicit.
        """
        predictions, cache = self.forward(x, return_cache=True)
        activations = cache["activations"]
        preactivations = cache["preactivations"]

        num_layers = len(self.weights)
        z_last = preactivations[-1]

        if loss_type == "cross_entropy":
            loss, grad_output = self._loss_and_output_grad(predictions, targets, loss_type, logits_for_ce=z_last)
        else:
            loss, grad_output = self._loss_and_output_grad(predictions, targets, loss_type)

        grad_weights = [None] * num_layers
        grad_biases = [None] * num_layers

        grad_a = grad_output

        for layer_index in reversed(range(num_layers)):
            a_prev = activations[layer_index]
            z = preactivations[layer_index]
            w = self.weights[layer_index]

            if layer_index == num_layers - 1:
                if self.output_activation in ("linear", "softmax"):
                    grad_z = grad_a
                else:
                    raise ValueError("Unsupported output activation in backward: %s" % self.output_activation)
            else:
                grad_activation = self._hidden_backward_derivative(z)
                grad_z = grad_a * grad_activation

            grad_w = np.matmul(a_prev.T, grad_z)
            grad_b = np.sum(grad_z, axis=0)

            grad_weights[layer_index] = grad_w
            grad_biases[layer_index] = grad_b

            grad_a = np.matmul(grad_z, w.T)

        return loss, grad_weights, grad_biases


    def apply_gradients(self, grad_weights, grad_biases, learning_rate=1e-3):
        """
        Apply a plain gradient-descent step in place. The method assumes gradients
        already have the same structure as the stored parameter lists.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]


    def predict(self, x):
        """
        Convenience inference wrapper. It intentionally does not build caches, which
        keeps execution lightweight when the model is only used for prediction.
        """
        return self.forward(x, return_cache=False)


    def save(self, path):
        """
        Persist architecture metadata and learned parameters into a NumPy archive.
        Activations are stored as small string arrays so the model can be rebuilt
        without depending on external config files.
        """
        data = { "layer_sizes": np.array(self.layer_sizes, dtype=np.int32),
            "hidden_activation": np.array([self.hidden_activation]).astype(np.bytes_),
            "output_activation": np.array([self.output_activation]).astype(np.bytes_)}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data["W_%d" % i] = w
            data["b_%d" % i] = b
        np.savez(path, **data)


    @staticmethod
    def load(path):
        """
        Reconstruct a previously saved MLP, then overwrite the freshly initialized
        random parameters with the serialized tensors from disk.
        """
        loaded = np.load(path, allow_pickle=True)
        layer_sizes = loaded["layer_sizes"].tolist()
        hidden_activation = loaded["hidden_activation"][0].decode()
        output_activation = loaded["output_activation"][0].decode()
        mlp = MLP(layer_sizes, hidden_activation=hidden_activation, output_activation=output_activation)
        num_layers = len(layer_sizes) - 1
        for i in range(num_layers):
            mlp.weights[i] = loaded[f"W_{i}"]
            mlp.biases[i] = loaded[f"b_{i}"]
        return mlp
