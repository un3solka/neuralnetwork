const {makeArray, arrayCopy} = require("./utils");
const Layer = require("./Layer");

class NeuralNetwork {
    constructor(learningRate, activation, derivative, ...sizes) {
        this.learningRate = learningRate;
        this.activation = activation;
        this.derivative = derivative;
        this.layers = new Layer(sizes.length);
        for (let i = 0; i < sizes.length; i++) {
            let nextSize = 0;
            if (i < sizes.length - 1) nextSize = sizes[i + 1];
            this.layers[i] = new Layer(sizes[i], nextSize);
            for (let j = 0; j < sizes[i]; j++) {
                this.layers[i].biases[j] = Math.random() * 2.0 - 1.0;
                for (let k = 0; k < nextSize; k++) {
                    this.layers[i].weights[j][k] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    feedForward(inputs) {
        arrayCopy(inputs, 0, this.layers[0].neurons, 0, inputs.length);
        for (let i = 1; i < this.layers.size; i++) {
            const l = this.layers[i - 1];
            const l1 = this.layers[i];
            for (let j = 0; j < l1.size; j++) {
                l1.neurons[j] = 0;
                for (let k = 0; k < l.size; k++) {
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                }
                l1.neurons[j] += l1.biases[j];
                l1.neurons[j] = this.activation(l1.neurons[j]);
            }
        }
        return this.layers[this.layers.size - 1].neurons;
    }

    backpropagation(targets) {
        let errors = Array(this.layers[this.layers.size - 1].size);
        for (let i = 0; i < this.layers[this.layers.size - 1].size; i++) {
            errors[i] = targets[i] - this.layers[this.layers.size - 1].neurons[i];
        }
        for (let k = this.layers.size - 2; k >= 0; k--) {
            const l = this.layers[k];
            const l1 = this.layers[k + 1];
            const errorsNext = Array(l.size);
            const gradients = Array(l1.size);
            for (let i = 0; i < l1.size; i++) {
                gradients[i] = errors[i] * this.derivative(this.layers[k + 1].neurons[i]);
                gradients[i] *= this.learningRate;
            }
            const deltas = makeArray(l1.size, l.size);
            for (let i = 0; i < l1.size; i++) {
                for (let j = 0; j < l.size; j++) {
                    deltas[i][j] = gradients[i] * l.neurons[j];
                }
            }
            for (let i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                for (let j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                }
            }
            errors = Array(l.size);
            arrayCopy(errorsNext, 0, errors, 0, l.size);
            const weightsNew = makeArray(l.weights.length, l.weights[0].length);
            for (let i = 0; i < l1.size; i++) {
                for (let j = 0; j < l.size; j++) {
                    weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                }
            }
            l.weights = weightsNew;
            for (let i = 0; i < l1.size; i++) {
                l1.biases[i] += gradients[i];
            }
        }
    }
}

module.exports = NeuralNetwork;