const {makeArray} = require("./utils");
class Layer {
    constructor(size, nextSize) {
        this.size = size;
        this.neurons = Array(size);
        this.biases = Array(size);
        this.weights = makeArray(size, nextSize);
    }
}

module.exports = Layer;