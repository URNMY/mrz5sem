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
    synapsesWeights.inputToHidden = getRandMatrix(inputLayerNeuronsCount, hiddenLayerNeuronsCount);
    synapsesWeights.hiddenLayer = getRandMatrix(hiddenLayerNeuronsCount, outputLayerNeuronsCount);
    synapsesWeights.noHiddenToHidden = getRandMatrix(contextNeuronsCount, hiddenLayerNeuronsCount);
    learningParameters = {
        currentError: 0,
        hiddenLayerPrejudices: new Array(hiddenLayerNeuronsCount).fill(0),
        limitError: Number(prompt('Input error limit:')),
        learningCoef: Number(prompt('Input learning coefficient'))
    }
    currentNotHiddenNeuronsValues = [new Array(contextNeuronsCount).fill(0)];
    isInitialized = true;
}

document.getElementById('button--reset').addEventListener('click', () => {
    init();
});

document.getElementById('button-predict').addEventListener('click', () => {
    calculateNetworkOutput();
});

function calculateNetworkOutput(inputValues, standardValue) {
    const input = inputValues ?? [prompt('Input 5 numbers').trim().split(' ').map(el => Number(el))];
    const standard = standardValue ?? Number(prompt('Input 6th expected value'));
    learningParameters.initSequence = input;
    learningParameters.currStandard = standard;

    const hiddenLayerNeuronValues =  sumMatrix(
        matrixMultiply(input, synapsesWeights.inputToHidden),
        matrixMultiply(currentNotHiddenNeuronsValues, synapsesWeights.noHiddenToHidden)
    ).map((row, indexRow) => row.map(el => hiddenNeuronActivationFunction(el, learningParameters.hiddenLayerPrejudices[indexRow])));
    const outputLayerNeuronValues = matrixMultiply(
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

document.getElementById('button-learn').addEventListener('click', () => {
    if (!predicted()) {
        alert('You need to predict at least once before learning');
        return;
    }
    learn();
});

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
    let k = 0;
    while (k < synapsesWeights.inputToHidden.length){
        let i = 0;
        while(i < differencingCoef.length) {
            synapsesWeights.inputToHidden[k][i] -= differencingCoef[i] * learningParameters.initSequence[0][k];
            i++;
        }
        k++;
    }
    let l = 0;
    while (l < synapsesWeights.noHiddenToHidden.length){
        let i = 0;
        while(i < differencingCoef.length) {
            synapsesWeights.noHiddenToHidden[l][i] -= differencingCoef[i] * learningParameters.hiddenLayerNeuronValues[i][0];
            i++;
        }
        k++;
    }
}

function tunePrejudices(differencingCoef) {
    learningParameters.hiddenLayerPrejudices = learningParameters.hiddenLayerPrejudices.map((prejudice, index) => prejudice + differencingCoef[index])
}

function predicted() {
    return isPredicted;
}

function matrixMultiply(A, B)
    {
        const rowsA = A.length, rowsB = B.length, columnsB = B[0].length, C = [];
        let i = 0;
        while (i < rowsA) C[i] = [];
        let k = 0;
        while (k < columnsB){
            let i = 0;
            while (i < rowsA){
                let newMatrixCell = 0;
                let j = 0;
                while(j < rowsB) {
                    newMatrixCell += A[i][j]*B[j][k];
                    j++;
                }
                C[i][k] = newMatrixCell;
                i++;
            }
            k++;
        }
        return C;
    }
    function getRandMatrix(height, width) {
        let bottomLine = -1, topLine = 1;
        const matrix = new Array(height).fill().map(el => new Array(width));
        let i = 0;
        while(i < height){
            let j = 0;
            while (j < width){
                matrix[i][j] = (topLine - bottomLine) * Math.random() + bottomLine;
                j++;
            }
            i++;
        }
        return matrix;
    }

    function sumMatrix(matrix1, matrix2) {
        const result = [];
        let i = 0;
        while (i < matrix1.length){
            result.push(new Array(matrix1[0].length));
            let j = 0;
            while (j < matrix1[0].length){
                result[i][j] = matrix1[i][j] + matrix2[i][j];
                j++;
            }
            i++;
        }
        return result;
    }



    

    

    


