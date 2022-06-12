const fs = require('fs');
const NeuralNetwork = require('./NeuralNetwork');
const Jimp = require('jimp');
const {makeArray} = require("./utils");
const path = "/Users/maksimkaronchyk/Downloads/train/";

async function digits() {
    const sigmoid = x => 1 / (1 + Math.exp(-x));
    const dsigmoid = y => y * (1 - y);
    const nn = new NeuralNetwork(0.001, sigmoid, dsigmoid, 784, 512, 128, 32, 10);

    const samples = 10;//60000
    const images = Array(samples);
    const digits = Array(samples);
    const imagesFiles = fs.readdirSync(path);
    for (let i = 0; i < samples; i++) {
        images[i] = fs.readFileSync(path + imagesFiles[i]);
        digits[i] = parseInt(imagesFiles[i].charAt(10) + "");
    }

    const image = await Jimp.read(images[0]);
    const inputs = makeArray(samples, 784);
    for (let i = 0; i < samples; i++) {
        for (let x = 0; x < 28; x++) {
            for (let y = 0; y < 28; y++) {
                const image = await Jimp.read(images[i]);
                const rgba = Jimp.intToRGBA(image.getPixelColor(x,y));
                const hex = rgba2hex(rgba);
                inputs[i][x + y * 28] = (Number('0x' + hex) & 0xff) / 255.0;
            }
        }
    }

    let epochs = 1000;
    for (let i = 1; i < epochs; i++) {
        let right = 0;
        let errorSum = 0;
        let batchSize = 100;
        for (let j = 0; j < batchSize; j++) {
            let imgIndex = Math.floor(Math.random() * samples);
            const targets = Array(10).fill(0);
            let digit = digits[imgIndex];
            targets[digit] = 1;

            const outputs = nn.feedForward(inputs[imgIndex]);
            let maxDigit = 0;
            let maxDigitWeight = -1;
            for (let k = 0; k < 10; k++) {
                if(outputs[k] > maxDigitWeight) {
                    maxDigitWeight = outputs[k];
                    maxDigit = k;
                }
            }
            if(digit === maxDigit) right++;
            for (let k = 0; k < 10; k++) {
                errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
            }
            nn.backpropagation(targets);
        }
        console.log("epoch: " + i + ". correct: " + right + ". error: " + errorSum);
    }

}

function rgba2hex({r, g, b, a}) {
    let hex =
        (r | 1 << 8).toString(16).slice(1) +
        (g | 1 << 8).toString(16).slice(1) +
        (b | 1 << 8).toString(16).slice(1);

    hex = (a | 1 << 8).toString(16).slice(1) + hex;
    return hex;
}

digits();