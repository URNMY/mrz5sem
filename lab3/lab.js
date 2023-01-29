const mathModule = (() => {
    function matrixMultiply(A, B)
    {
        const rowsA = A.length,
            rowsB = B.length, columnsB = B[0].length,
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

    function differenceMatrices(matrix1, matrix2) {
        const result = [];
        const height = matrix1.length;
        const width = matrix1[0].length;
        for (let i = 0; i < height; i++) {
            result.push(new Array(width));
            for(let j = 0; j < width; j++) {
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
        return result;
    }


    function multiplyMatrixByNumber(matrix, number) {
        const newMatrix = new Array(matrix.length).fill().map(el => new Array(matrix[0].length));
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[0].length; j++) {
                newMatrix[i][j] = matrix[i][j] * number;
            }
        }
        return newMatrix;
    }

    function getRandMatrix(height, width) {
        const bottomLine = -1, topLine = 1;
        const matrix = new Array(height).fill().map(el => new Array(width));
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                matrix[i][j] = randNumberInRange(bottomLine, topLine);
            }
        }
        return matrix;
    }

    function randNumberInRange(leftBound, rightBound) {
        return (rightBound - leftBound) * Math.random() + leftBound;
    }

    function functionValuesToMatrixElems(matrix, func) {
        return matrix.map(row => row.map(el => func(el)));
    }

    function sumMatrix(matrix1, matrix2) {
        return differenceMatrices(matrix1, multiplyMatrixByNumber(matrix2, -1));
    }
    return {
        matrixMultiply,
        getRandMatrix,
        functionValuesToMatrixElems,
        sumMatrix
    }
})();

const leakyReLUModule = (() => {
    const contextNeuronActivationFunction = Math.asinh;
    const hiddenNeuronActivationFunction = (arg) => arg;
    const outputNeuronActivationFunction = (arg) => arg;
    const hiddenNeuronActivationDerivativeFunction = (arg) => 1;
    const inputLayerNeuronsQuantity = 5;
    const hiddenLayerNeuronsQuantity = 1;
    const outputLayerNeuronsQuantity = 1;
    const contextNeuronsQuantity = outputLayerNeuronsQuantity;
    const weights = {};
    let learningParameters = {}
    let currentNotHiddenNeuronsValues;
    let initialized = false;
    let predictedOnce = false;

    function init() {
        weights.inputToHidden = mathModule.getRandMatrix(inputLayerNeuronsQuantity, hiddenLayerNeuronsQuantity);
        weights.hiddenLayer = mathModule.getRandMatrix(hiddenLayerNeuronsQuantity, outputLayerNeuronsQuantity);
        weights.noHiddenToHidden = mathModule.getRandMatrix(contextNeuronsQuantity, hiddenLayerNeuronsQuantity);
        learningParameters = {
            currentError: 0,
            hiddenLayerPrejudices: new Array(hiddenLayerNeuronsQuantity).fill(0),
            maxError: Number(prompt('Input max error:')),
            learningCoef: Number(prompt('Input learning coefficient'))
        }
        currentNotHiddenNeuronsValues = [new Array(contextNeuronsQuantity).fill(0)];
        initialized = true;
    }

    function calculateNetworkOutput(inputValues, standardValue) {
        const input = inputValues ?? [prompt('Input 5 numbers').trim().split(' ').map(el => Number(el))];
        const standard = standardValue ?? Number(prompt('Input 6th expected value'));
        learningParameters.initSequence = input;
        learningParameters.currStandard = standard;

        const hiddenLayerNeuronValues =  mathModule.sumMatrix(
            mathModule.matrixMultiply(input, weights.inputToHidden),
            mathModule.matrixMultiply(currentNotHiddenNeuronsValues, weights.noHiddenToHidden)
        ).map((row, indexRow) => row.map(el => hiddenNeuronActivationFunction(el, learningParameters.hiddenLayerPrejudices[indexRow])));
        const outputLayerNeuronValues = mathModule.functionValuesToMatrixElems(
            mathModule.matrixMultiply(
                hiddenLayerNeuronValues,
                weights.hiddenLayer
            ),
            outputNeuronActivationFunction
        );
        currentNotHiddenNeuronsValues = mathModule.functionValuesToMatrixElems(
            outputLayerNeuronValues,
            contextNeuronActivationFunction
        );

        learningParameters.hiddenLayerNeuronValues = hiddenLayerNeuronValues;
        learningParameters.outputLayerNeuronValues = outputLayerNeuronValues;
        learningParameters.currentError = outputLayerNeuronValues[0][0] - standard;
        predictedOnce = true;
        console.log(outputLayerNeuronValues[0][0]);
    }

    function learn() {
        while (Math.abs(learningParameters.currentError) > learningParameters.maxError) {
            tuneWeights();
            tunePrejudices(learningParameters.differencingCoef);
            calculateNetworkOutput(learningParameters.initSequence, learningParameters.currStandard);
        }
    }

    function tuneWeights() {
        weights.hiddenLayer = weights.hiddenLayer.map((row, indexRow) => {
            const differencing = learningParameters.learningCoef * learningParameters.currentError * learningParameters.hiddenLayerNeuronValues[indexRow][0];
            return row.map(el => el - differencing);
        });

        learningParameters.differencingCoef = learningParameters.hiddenLayerNeuronValues.map((row, indexRow) =>
            learningParameters.learningCoef * ithGamma(learningParameters.currentError, indexRow) * hiddenNeuronActivationDerivativeFunction(row[0])
        );

        const differencingCoef = learningParameters.differencingCoef;

        for (let k = 0; k < weights.inputToHidden.length; k++) {
            for (let i = 0; i < differencingCoef.length; i++) {
                weights.inputToHidden[k][i] -= differencingCoef[i] * learningParameters.initSequence[0][k];
            }
        }

        for (let l = 0; l < weights.noHiddenToHidden.length; l++) {
            for (let i = 0; i < differencingCoef.length; i++) {
                weights.noHiddenToHidden[l][i] -= differencingCoef[i] * learningParameters.hiddenLayerNeuronValues[i][0];
            }
        }
    }

    function tunePrejudices(differencingCoef) {
        learningParameters.hiddenLayerPrejudices = learningParameters.hiddenLayerPrejudices.map((prejudice, index) => prejudice + differencingCoef[index])
    }

    function ithGamma(currentError, index) {
        return currentError * weights.hiddenLayer[index][0];
    }

    function isInitialized() {
        return initialized;
    }

    function predicted() {
        return predictedOnce;
    }

    return {
        init,
        calculateNetworkOutput,
        learn,
        isInitialized,
        predicted
    }
})();

document.getElementById('init-reset').addEventListener('click', () => {
    leakyReLUModule.init();
});

document.getElementById('predict').addEventListener('click', () => {
    if (!leakyReLUModule.isInitialized()) {
        alert('Network is not initialized!');
        return;
    }
    leakyReLUModule.calculateNetworkOutput();
});

document.getElementById('learn').addEventListener('click', () => {
    if (!leakyReLUModule.isInitialized()) {
        alert('Network is not initialized!');
        return;
    }
    if (!leakyReLUModule.predicted()) {
        alert('You need to predict at least once before learning');
        return;
    }
    leakyReLUModule.learn();
});
