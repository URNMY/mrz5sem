const mathModule = (() => {
    function matrixMultiply(A, B)
    {
        const rowsA = A.length, rowsB = B.length, columnsB = B[0].length,
            C = [];
        for (let i = 0; i < rowsA; i++) C[i] = [];
        for (let k = 0; k < columnsB; k++) {
            for (let i = 0; i < rowsA; i++) {
                let newMatrixCell = 0;
                for (let j = 0; j < rowsB; j++) newMatrixCell += A[i][j]*B[j][k];
                C[i][k] = newMatrixCell;
            }
        }
        return C;
    }

    function getRandMatrix(height, width) {
        let bottomLine = -1, topLine = 1;
        const matrix = new Array(height).fill().map(el => new Array(width));
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                matrix[i][j] = (topLine - bottomLine) * Math.random() + bottomLine;

            }
        }
        return matrix;
    }

    function sumMatrix(matrix1, matrix2) {
        const result = [];
        for (let i = 0; i < matrix1.length; i++) {
            result.push(new Array(matrix1[0].length));
            for(let j = 0; j < matrix1[0].length; j++) {
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return result;
    }
    return {
        matrixMultiply,
        getRandMatrix,
        sumMatrix
    }
})();

const leakyReLUModule = (() => {
    const contextNeuronActivationFunction = Math.asinh;
    const hiddenNeuronActivationFunction = (arg) => arg;
    const outputNeuronActivationFunction = (arg) => arg;
    const hiddenNeuronActivationDerivativeFunction = () => 1;
    const inputLayerNeuronsCount = 5;
    const hiddenLayerNeuronsCount = 1;
    const outputLayerNeuronsCount = 1;
    const contextNeuronsCount = outputLayerNeuronsCount;
    const synapsesWeights = {};
    let learningParameters = {}
    let currentNotHiddenNeuronsValues;
    let isInitialized = false;
    let isPredicted = false;

    function init() {
        synapsesWeights.inputToHidden = mathModule.getRandMatrix(inputLayerNeuronsCount, hiddenLayerNeuronsCount);
        synapsesWeights.hiddenLayer = mathModule.getRandMatrix(hiddenLayerNeuronsCount, outputLayerNeuronsCount);
        synapsesWeights.noHiddenToHidden = mathModule.getRandMatrix(contextNeuronsCount, hiddenLayerNeuronsCount);
        learningParameters = {
            currentError: 0,
            hiddenLayerPrejudices: new Array(hiddenLayerNeuronsCount).fill(0),
            limitError: Number(prompt('Input error limit:')),
            learningCoef: Number(prompt('Input learning coefficient'))
        }
        currentNotHiddenNeuronsValues = [new Array(contextNeuronsCount).fill(0)];
        isInitialized = true;
    }

    function calculateNetworkOutput(inputValues, standardValue) {
        const input = inputValues ?? [prompt('Input 5 numbers').trim().split(' ').map(el => Number(el))];
        const standard = standardValue ?? Number(prompt('Input 6th expected value'));
        learningParameters.initSequence = input;
        learningParameters.currStandard = standard;

        const hiddenLayerNeuronValues =  mathModule.sumMatrix(
            mathModule.matrixMultiply(input, synapsesWeights.inputToHidden),
            mathModule.matrixMultiply(currentNotHiddenNeuronsValues, synapsesWeights.noHiddenToHidden)
        ).map((row, indexRow) => row.map(el => hiddenNeuronActivationFunction(el, learningParameters.hiddenLayerPrejudices[indexRow])));
        const outputLayerNeuronValues = mathModule.matrixMultiply(
                hiddenLayerNeuronValues,
                synapsesWeights.hiddenLayer
                ).map(row => row.map(el => outputNeuronActivationFunction(el)));
        currentNotHiddenNeuronsValues = outputLayerNeuronValues.map(row => row.map(el => contextNeuronActivationFunction(el)));

        learningParameters.hiddenLayerNeuronValues = hiddenLayerNeuronValues;
        learningParameters.outputLayerNeuronValues = outputLayerNeuronValues;
        learningParameters.currentError = outputLayerNeuronValues[0][0] - standard;
        isPredicted = true;
        console.log(outputLayerNeuronValues[0][0]);
    }

    function learn() {
        while (Math.abs(learningParameters.currentError) > learningParameters.limitError) {
            tuneWeights();
            tunePrejudices(learningParameters.differencingCoef);
            calculateNetworkOutput(learningParameters.initSequence, learningParameters.currStandard);
        }
    }

    function tuneWeights() {
        synapsesWeights.hiddenLayer = synapsesWeights.hiddenLayer.map((row, indexRow) => {
            const differencing = learningParameters.learningCoef * learningParameters.currentError * learningParameters.hiddenLayerNeuronValues[indexRow][0];
            return row.map(el => el - differencing);
        });

        learningParameters.differencingCoef = learningParameters.hiddenLayerNeuronValues.map((row, indexRow) =>
            learningParameters.learningCoef * learningParameters.currentError * synapsesWeights.hiddenLayer[indexRow][0] * hiddenNeuronActivationDerivativeFunction()
        );

        const differencingCoef = learningParameters.differencingCoef;

        for (let k = 0; k < synapsesWeights.inputToHidden.length; k++) {
            for (let i = 0; i < differencingCoef.length; i++) {
                synapsesWeights.inputToHidden[k][i] -= differencingCoef[i] * learningParameters.initSequence[0][k];
            }
        }

        for (let l = 0; l < synapsesWeights.noHiddenToHidden.length; l++) {
            for (let i = 0; i < differencingCoef.length; i++) {
                synapsesWeights.noHiddenToHidden[l][i] -= differencingCoef[i] * learningParameters.hiddenLayerNeuronValues[i][0];
            }
        }
    }

    function tunePrejudices(differencingCoef) {
        learningParameters.hiddenLayerPrejudices = learningParameters.hiddenLayerPrejudices.map((prejudice, index) => prejudice + differencingCoef[index])
    }

    function predicted() {
        return isPredicted;
    }

    return {
        init,
        calculateNetworkOutput,
        learn,
        predicted
    }
})();

document.getElementById('button--reset').addEventListener('click', () => {
    leakyReLUModule.init();
});

document.getElementById('button-predict').addEventListener('click', () => {
    leakyReLUModule.calculateNetworkOutput();
});

document.getElementById('button-learn').addEventListener('click', () => {
    if (!leakyReLUModule.predicted()) {
        alert('You need to predict at least once before learning');
        return;
    }
    leakyReLUModule.learn();
});
