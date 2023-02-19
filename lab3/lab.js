const contextNeuronActivationFunction = Math.asinh,
 hiddenNeuronActivationFunction = (arg) => arg,
 outputNeuronActivationFunction = (arg) => arg,
 hiddenNeuronActivationDerivativeFunction = () => 1,
 inputLayerNeuronsCount = 5,
 hiddenLayerNeuronsCount = 1,
 outputLayerNeuronsCount = 1,
 contextNeuronsCount = outputLayerNeuronsCount,
 synapsesWeights = {};
let learningParameters = {},
 currentNotHiddenNeuronsValues;

document.getElementById('button--reset').addEventListener('click', () => {
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

});

document.getElementById('button-predict').addEventListener('click', () => {
    calculateNetworkOutput();
});

function calculateNetworkOutput(inputValues, standardValue) {
    const input = inputValues ?? [prompt('Input 5 numbers').trim().split(' ').map(el => Number(el))],
     standard = standardValue ?? Number(prompt('Input 6th expected value'));
    learningParameters.initSequence = input;
    learningParameters.currStandard = standard;
    const hiddenLayerNeuronValues =  sumMatrix(
        matrixMultiply(input, synapsesWeights.inputToHidden),
        matrixMultiply(currentNotHiddenNeuronsValues, synapsesWeights.noHiddenToHidden)
    ).map((row, indexRow) => row.map(el => hiddenNeuronActivationFunction(el, learningParameters.hiddenLayerPrejudices[indexRow]))),
     outputLayerNeuronValues = matrixMultiply(
        hiddenLayerNeuronValues,
        synapsesWeights.hiddenLayer
    ).map(row => row.map(el => outputNeuronActivationFunction(el)));
    currentNotHiddenNeuronsValues = outputLayerNeuronValues.map(row => row.map(el => contextNeuronActivationFunction(el)));
    learningParameters.hiddenLayerNeuronValues = hiddenLayerNeuronValues;
    learningParameters.outputLayerNeuronValues = outputLayerNeuronValues;
    learningParameters.currentError = outputLayerNeuronValues[0][0] - standard;
    console.log(outputLayerNeuronValues[0][0]);
}

document.getElementById('button-learn').addEventListener('click', () => {
    while (Math.abs(learningParameters.currentError) > learningParameters.limitError) {
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
                synapsesWeights.inputToHidden[k][i] = synapsesWeights.inputToHidden[k][i] - differencingCoef[i] * learningParameters.initSequence[0][k];
                i++;
            }
            k++;
        }
        let l = 0;
        while (l < synapsesWeights.noHiddenToHidden.length){
            let i = 0;
            while(i < differencingCoef.length) {
                synapsesWeights.noHiddenToHidden[l][i] = synapsesWeights.noHiddenToHidden[l][i] - differencingCoef[i] * learningParameters.hiddenLayerNeuronValues[i][0];
                i++;
            }
            k++;
        }
        learningParameters.hiddenLayerPrejudices = learningParameters.hiddenLayerPrejudices.map((prejudice, index) => prejudice + learningParameters.differencingCoef[index])
        calculateNetworkOutput(learningParameters.initSequence, learningParameters.currStandard);
    }
});

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
                    newMatrixCell = newMatrixCell + A[i][j] * B[j][k];
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
        const matrix = new Array(height).map(el => new Array(width));
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
