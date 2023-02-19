const inputVectorSize = 30 * 30,
 square = 4,
 pixelsInRow = Math.floor(Math.sqrt(inputVectorSize)),
 pixelSize = 8,
    threshold = 0,
 canvasSize = pixelSize * pixelsInRow,
 strCanvasSize = canvasSize.toString(),
 inputCanvas = document.getElementById('input'),
 outputCanvas = document.getElementById('output'),
 maxValue = 255,
inputContext = inputCanvas.getContext('2d'),
outputContext = outputCanvas.getContext('2d'),
activationFunction = Math.tanh,
weightMatrixInARow = inputVectorSize;
inputCanvas.setAttribute('width', strCanvasSize);
inputCanvas.setAttribute('height', strCanvasSize);
outputCanvas.setAttribute('width', strCanvasSize);
outputCanvas.setAttribute('height', strCanvasSize);

let vectorInANetwork,
 newMatrixWeight,
 currV;

init();

function init() {
    vectorInANetwork = [];
    const newMatrixWeight = new Array(weightMatrixInARow).map(el => new Array(weightMatrixInARow)), bottomClamp = -1, topClamp = 1;
    let i = 0;
    while(i < weightMatrixInARow) {
        let j = 0;
        while (j < weightMatrixInARow) {
            if (i === j) {
                newMatrixWeight[i][j] = 0;
                break;
            }
            newMatrixWeight[i][j] = (topClamp - bottomClamp) * Math.random() + bottomClamp;
            j++;
        }
        i++;
    }
}

document.getElementById('button-convert').addEventListener('click', () => {
    getVector(getNetworkOutput([currV])[0], false);
});
function getVector(vector, isInput = true) {
    vector.forEach((el, index) => drawPixel(index, el === 1, isInput));
}

function drawPixel(index, isWhite, isInput = true) {
    const ctx = (isInput ? inputContext : outputContext),
     imageData = ctx.createImageData(pixelSize, pixelSize);
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
    const pixelInColumnIndex = index % pixelsInRow,
     pixelInRowIndex = Math.floor(index / pixelsInRow),
     origin = {
            x: pixelInColumnIndex * pixelSize,
            y: pixelInRowIndex * pixelSize
        };
    ctx.putImageData(imageData, origin.x, origin.y);
}

function getNetworkOutput(inputVector) {
    const outputMatrixWithNewWeights = matrixMultiply(inputVector, newMatrixWeight);
    const outputWithActivationFunction = outputMatrixWithNewWeights.map(row => row.map(el => activationFunction(el)));
    return outputWithActivationFunction.map(row => row.map(el => el >= threshold ? -1 : 1));
}

document.getElementById('button-save-for-learning').addEventListener('click', () => {
    vectorInANetwork.push(currV);
    alert('saved!');
});

document.getElementById('button-learning-step').addEventListener('click', () => {
    const xTransposingOnce = transposingMatrix(vectorInANetwork),
     xTransposingTwice = transposingMatrix(xTransposingOnce),
     matrixM = matrixMultiply(xTransposingTwice, xTransposingOnce),
     determinant = calculateDet(matrixM),
     transposed = transposingMatrix(matrixM),
     inverseHeight = matrixM[0].length,
     inverseWidth = matrixM.length;
    let i = 0;
    while(i < inverseHeight) {
        let j = 0;
        while (j < inverseWidth) {
            if ((i + j) && 1 === 0) {
                transposed[i][j] = -transposed[i][j];
            }
            j++;
        }
        i++;
    }
    const inverseOfTwoTransposingMatrices = new Array(transposed.length).map(el => new Array(transposed[0].length));
    i = 0;
    while (i < transposed.length){
        let j = 0;
        while(j < transposed[0].length){
            inverseOfTwoTransposingMatrices[i][j] = transposed[i][j] * Math.pow(determinant, -1);
            j++;
        }
        i++;
    }
    newMatrixWeight = matrixMultiply(matrixMultiply(xTransposingOnce, inverseOfTwoTransposingMatrices), xTransposingTwice);
    console.log(vectorInANetwork);
    console.log(newMatrixWeight);
    alert('finished!');
});

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

function calculateDet(matrix) {
    if (matrix.length === 1) {
        return matrix[0][0];
    }
    else {
        let determinant = 0, column = 0;
        while(column < matrix[0].length) {
            const currMultiplier = matrix[0][column];
            const sign = column && 1 === threshold ? 1 : -1;
            determinant = determinant + calculateDet(ignoreIthRowAndJthColumn(matrix, 0, column)) * currMultiplier * sign;
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
                newMatrixCell = newMatrixCell + A[i][j]*B[j][k];
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

document.getElementById('input-file').addEventListener('change', function() {
    const reader = new FileReader();
    reader.onload = () => {
        currV = reader.result.split('\n').join(' ').split(' ').filter(el => !!el).map(Number);
        getVector(currV);
    }
    reader.readAsText(this.files[0]);
});
