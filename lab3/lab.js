const matrixModule = (() => {
    function matrixMultiply(A, B)
    {
        const rowsA = A.length, columnsA = A[0].length,
            rowsB = B.length, columnsB = B[0].length,
            C = [];

        try {
            if (columnsA !== rowsB) {
                throw `Invalid matrix size; A length = ${A[0].length}, B height = ${B.length}`;
            }
        }
        catch(err) {
            throw err;
        }

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

    function normalizeRow(matrix, indexRow) {
        const modulus = Math.sqrt(matrix[indexRow].map(el => el * el).reduce((r, v) => r + v, 0));
        for(let j = 0; j < matrix[indexRow].length; j++) {
            matrix[indexRow][j] /= modulus;
        }
    }

    function ignoreIthRowAndJthColumn(matrix, i, j) {
        const newMatrix = [];
        let skippedRow = false;
        for (let row = 0; row < matrix.length; row++) {
            if (row === i) {
                skippedRow = true;
                continue;
            }
            newMatrix.push([]);
            for (let column = 0; column < matrix[0].length; column++) {
                if (column == j) {
                    continue;
                }
                newMatrix[row - skippedRow].push(matrix[row][column]);
            }
        }
        return newMatrix;
    }

    function getRandMatrix(height, width, nullificateMainDiagonal = true, bottomClamp = -1, topClamp = 1) {
        const matrix = new Array(height).fill().map(el => new Array(width));
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                if (nullificateMainDiagonal && i === j) {
                    matrix[i][j] = 0;
                    continue;
                }
                matrix[i][j] = randNumberInRange(bottomClamp, topClamp);
            }
        }
        return matrix;
    }

    function randNumberInRange(leftBound, rightBound) {
        return (rightBound - leftBound) * Math.random() + leftBound;
    }

    function applyFunctionToMatrixElements(matrix, func) {
        return matrix.map(row => row.map(el => func(el)));
    }

    function sumMatrix(matrix1, matrix2) {
        return differenceMatrices(matrix1, multiplyMatrixByNumber(matrix2, -1));
    }
    return {
        matrixMultiply,
        getRandMatrix,
        applyFunctionToMatrixElements,
        sumMatrix
    }
})();

const lrluNetworkModule = (() => {
    const contextNeuronActivationFunction = Math.asinh;
    const hiddenNeuronActivationFunction = (arg) => arg;
    const outputNeuronActivationFunction = (arg) => arg;
    const hiddenNeuronActivationFunctionDerivative = (arg) => 1;
    const inputNeuronsQuantity = 5;
    const hiddenNeuronsQuantity = 1;
    const outputNeuronsQuantity = 1;
    const contextNeuronsQuantity = outputNeuronsQuantity;
    const weights = {};
    let learningParameters = {}
    let currContextNeuronsValues;

    let initialized = false;
    let predictedOnce = false;

    function init() {
        weights.inputToHidden = matrixModule.getRandMatrix(inputNeuronsQuantity, hiddenNeuronsQuantity, false);
        weights.hiddenOut = matrixModule.getRandMatrix(hiddenNeuronsQuantity, outputNeuronsQuantity, false);
        weights.contextToHidden = matrixModule.getRandMatrix(contextNeuronsQuantity, hiddenNeuronsQuantity, false);
        learningParameters = {
            currentError: 0,
            hiddenLayerPrejudices: new Array(hiddenNeuronsQuantity).fill(0),
            maxError: Number(prompt('Input max error:')),
            learningCoef: Number(prompt('Input learning coefficient'))
        }
        currContextNeuronsValues = [new Array(contextNeuronsQuantity).fill(0)];
        initialized = true;
    }

    function calculateNetworkOutput(inputValues, ethaloneValue) {


        const input = inputValues ?? [prompt('Input sequence (5 numbers, delimit by space)').trim().split(' ').map(el => Number(el))];
        const ethalone = ethaloneValue ?? Number(prompt('Input ethalone (1 number)'));
        learningParameters.currentIn = input;
        learningParameters.currEthalone = ethalone;

        const hiddenNeuronValues =  matrixModule.sumMatrix(
            matrixModule.matrixMultiply(input, weights.inputToHidden),
            matrixModule.matrixMultiply(currContextNeuronsValues, weights.contextToHidden)
        ).map((row, indexRow) => row.map(el => hiddenNeuronActivationFunction(el, learningParameters.hiddenLayerPrejudices[indexRow])));
        const outputNeuronValues = matrixModule.applyFunctionToMatrixElements(
            matrixModule.matrixMultiply(
                hiddenNeuronValues,
                weights.hiddenOut
            ),
            outputNeuronActivationFunction
        );
        currContextNeuronsValues = matrixModule.applyFunctionToMatrixElements(
            outputNeuronValues,
            contextNeuronActivationFunction
        );

        learningParameters.hiddenNeuronValues = hiddenNeuronValues;
        learningParameters.outputNeuronValues = outputNeuronValues;
        learningParameters.currentError = outputNeuronValues[0][0] - ethalone;
        predictedOnce = true;
        console.log(outputNeuronValues[0][0]);
    }

    function learn() {
        while (Math.abs(learningParameters.currentError) > learningParameters.maxError) {
            tuneWeights();
            tunePrejudices(learningParameters.differencingCoef);
            calculateNetworkOutput(learningParameters.currentIn, learningParameters.currEthalone);
        };
    }

    function tuneWeights() {
        weights.hiddenOut = weights.hiddenOut.map((row, indexRow) => {
            const differencing = learningParameters.learningCoef * learningParameters.currentError * learningParameters.hiddenNeuronValues[indexRow][0];
            return row.map(el => el - differencing);
        });

        learningParameters.differencingCoef = learningParameters.hiddenNeuronValues.map((row, indexRow) =>
            learningParameters.learningCoef
            * ithGamma(learningParameters.currentError, indexRow)
            * hiddenNeuronActivationFunctionDerivative(row[0])
        );

        const differencingCoef = learningParameters.differencingCoef;

        for (let k = 0; k < weights.inputToHidden.length; k++) {
            for (let i = 0; i < differencingCoef.length; i++) {
                weights.inputToHidden[k][i] -= differencingCoef[i] * learningParameters.currentIn[0][k];
            }
        }

        for (let l = 0; l < weights.contextToHidden.length; l++) {
            for (let i = 0; i < differencingCoef.length; i++) {
                weights.contextToHidden[l][i] -= differencingCoef[i] * learningParameters.hiddenNeuronValues[i][0];
            }
        }
    }

    function tunePrejudices(differencingCoef) {
        learningParameters.hiddenLayerPrejudices = learningParameters.hiddenLayerPrejudices.map((prejudice, index) => prejudice + differencingCoef[index])
    }

    function ithGamma(currentError, index) {
        return currentError * weights.hiddenOut[index][0];
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
    lrluNetworkModule.init();
});

document.getElementById('predict').addEventListener('click', () => {
    if (!lrluNetworkModule.isInitialized()) {
        alert('network not yet initialized');
        return;
    }
    lrluNetworkModule.calculateNetworkOutput();
});

document.getElementById('learn').addEventListener('click', () => {
    if (!lrluNetworkModule.isInitialized()) {
        alert('network not yet initialized');
        return;
    }
    if (!lrluNetworkModule.predicted()) {
        alert('you need to predict at least once before learning');
        return;
    }
    lrluNetworkModule.learn();
});
