const inputVectorSize = 30 * 30;
const square = 4;
const pixelsInRow = Math.floor(Math.sqrt(inputVectorSize));
const pixelSize = 8;
const canvasSize = pixelSize * pixelsInRow;
const strCanvasSize = canvasSize.toString();
const inputCanvas = document.getElementById('input');
const outputCanvas = document.getElementById('output');
const maxValue = 255;
inputCanvas.setAttribute('width', strCanvasSize);
inputCanvas.setAttribute('height', strCanvasSize);
outputCanvas.setAttribute('width', strCanvasSize);
outputCanvas.setAttribute('height', strCanvasSize);
const inputContext = inputCanvas.getContext('2d');
const outputContext = outputCanvas.getContext('2d');
const activationFunction = Math.tanh;
const weightMatrixInARow = inputVectorSize;
let vectorInANetwork;
let newMatrixWeight;
let currV;

init();

function init() {
    vectorInANetwork = [];
    newMatrixWeight = getRandMatrix(weightMatrixInARow, weightMatrixInARow);
}

function getRandMatrix(height, width, revokeMainDiagonal = true, bottomClamp = -1, topClamp = 1) {
    const matrix = new Array(height).fill().map(el => new Array(width));
    let i = 0;
    while(i < height) {
        let j = 0;
        while (j < width) {
            if (revokeMainDiagonal && i === j) {
                matrix[i][j] = 0;
                break;
            }
            matrix[i][j] = (topClamp - bottomClamp) * Math.random() + bottomClamp;
            j++;
        }
        i++;
    }
    return matrix;
}

document.getElementById('button-convert').addEventListener('click', () => {
    getVector(getNetworkOutput([currV])[0], false);
});
function getVector(vector, isInput = true) {
    if (vector.length !== Math.pow(pixelsInRow, 2)) return;
    vector.forEach((el, index) => drawPixel(index, el === 1, isInput));
}

function drawPixel(index, isWhite, isInput = true) {
    const ctx = (isInput ? inputContext : outputContext);
    const imageData = ctx.createImageData(pixelSize, pixelSize);
    let pixByTwo = 0;
    while(pixByTwo < Math.pow(pixelSize, 2)) {
        const colorValue = isWhite ? maxValue : 0;
        for (let i = 0; i < square; i++){
            if (i !== square - 1){
                imageData.data[i + square * pixByTwo] = colorValue;
            }
            else {
                imageData.data[i + square * pixByTwo] = maxValue;
            }
        }
        pixByTwo++;
    }
    const origin = getPixelOrigin(index);
    ctx.putImageData(imageData, origin.x, origin.y);
}

function getPixelOrigin(index, size = pixelSize) {
    const pixelInColumnIndex = index % pixelsInRow;
    const pixelInRowIndex = Math.floor(index / pixelsInRow);
    return {
        x: pixelInColumnIndex * size,
        y: pixelInRowIndex * size
    };
}

function getNetworkOutput(inputVector) {
    const outputMatrixWithNewWeights = matrixMultiply(inputVector, newMatrixWeight);
    const outputWithActivationFunction = outputMatrixWithNewWeights.map(row => row.map(el => activationFunction(el)));
    return outputWithActivationFunction.map(row => row.map(el => el >= 0 ? -1 : 1));
}

document.getElementById('button-save-for-learning').addEventListener('click', () => {
    if (!currV) return;
    vectorInANetwork.push(currV);
    alert('saved!');
});

document.getElementById('button-learning-step').addEventListener('click', () => {
    learningStep();
    alert('finished!');
});

function learningStep() {
    if (!vectorInANetwork.length) return;
    const xTransposingOnce = transposingMatrix(vectorInANetwork);
    const xTransposingTwice = transposingMatrix(xTransposingOnce);
    const inverseOfTwoTransposingMatrices = inverseMatrix(matrixMultiply(xTransposingTwice, xTransposingOnce));
    newMatrixWeight = matrixMultiply(matrixMultiply(xTransposingOnce, inverseOfTwoTransposingMatrices), xTransposingTwice);
    console.log(vectorInANetwork);
    console.log(newMatrixWeight);
}

function transposingMatrix(matrix) {
    const transposed = [];
    let j = 0;
    while (j < matrix[0].length){
        transposed.push(new Array(matrix.length));
        let i = 0;
        while(i < matrix.length) {
            transposed[j][i] = matrix[i][j];
            i++;
        }
        j++;
    }
    return transposed;
}

function inverseMatrix(matrix) {
    const determinant = calculateDet(matrix);
    const transposed = transposingMatrix(matrix);
    const inverseHeight = matrix[0].length;
    const inverseWidth = matrix.length;
    let i = 0;
    while(i < inverseHeight) {
        let j = 0;
        while (j < inverseWidth) {
            if ((i + j) && 1 === 0) {
                transposed[i][j] *= -1;
            }
            j++;
        }
        i++;
    }
    return multiplyMatrixByNumber(transposed, 1 / determinant);
}

function multiplyMatrixByNumber(matrix, number) {
    const newMatrix = new Array(matrix.length).fill().map(el => new Array(matrix[0].length));
    let i = 0;
    while (i < matrix.length){
        let j = 0;
        while(j < matrix[0].length){
            newMatrix[i][j] = matrix[i][j] * number;
            j++;
        }
        i++;
    }
    return newMatrix;
}
function calculateDet(matrix) {
    if (matrix.length === 1) {
        return matrix[0][0];
    }
    else {
        let determinant = 0, column = 0;
        while(column < matrix[0].length) {
            const currMultiplier = matrix[0][column];
            const sign = column && 1 === 0 ? 1 : -1;
            determinant += calculateDet(ignoreIthRowAndJthColumn(matrix, 0, column)) * currMultiplier * sign;
            column++;
        }
        return determinant;
    }
}

function ignoreIthRowAndJthColumn(matrix, i, j) {
    const newMatrix = [];
    let skippedRow = false;
    let row = 0;
    while(row < matrix.length) {
        if (row === i) {
            skippedRow = true;
            break;
        }
        newMatrix.push([]);
        let column = 0;
        while(column < matrix[0].length){
            if (column === j) {
                break;
            }
            newMatrix[row - skippedRow].push(matrix[row][column]);
            column++;
        }
        row++;
    }
    return newMatrix;
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

document.getElementById('button-reset').addEventListener('click', () => {
    init();
});

document.getElementById('text-file-input').addEventListener('change', function() {
    const reader = new FileReader();
    reader.onload = () => {
        currV = convertStringToVector(reader.result);
        getVector(currV);
    }
    reader.readAsText(this.files[0]);
});

function convertStringToVector(string) {
    return string.split('\n').join(' ').split(' ').filter(el => !!el).map(Number);
}
