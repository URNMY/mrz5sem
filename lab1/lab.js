document.getElementById('compress-button').addEventListener('click', () => {
    try {
        compressAndDecompress(learningParameters);
    }
    catch(err) {
        console.error(err);
        return;
    }
});

function compressAndDecompress(learningParameters) {
    learningParameters.parameters = [];
    const weights = learningParameters.currWeights;
    for(let subImageInColumnIndex = 0; subImageInColumnIndex < subImagesInOneColumn; subImageInColumnIndex++) {
        for(let subImageInRowIndex = 0; subImageInRowIndex < subImagesInOneRow; subImageInRowIndex++) {
            const origin = getOrigin(subImageInColumnIndex, subImageInRowIndex);
            const currInput = getNeuralNetworkSubImageInput(subImageInColumnIndex, subImageInRowIndex);
            const secondLayerValues = neuralNetworkModule.matrixMultiply(currInput, weights[0]);
            const currOutput = neuralNetworkModule.matrixMultiply(secondLayerValues, weights[1]);
            drawSubImage(
                currOutput[0].map(valueColorMapper),
                compressedImageCtx,
                origin.x,
                origin.y
            )
            learningParameters.parameters.push({
                currInput,
                secondLayerValues,
                currOutput,
            });
        }
    }
}

function getOrigin(subImageInColumnIndex, subImageInRowIndex, width = subImageWidth, height = subImageHeight) {
    return {
        x: subImageInRowIndex * width,
        y: subImageInColumnIndex * height
    };
}

function getNeuralNetworkSubImageInput(subImageInColumnIndex, subImageInRowIndex) {
    const origin = getOrigin(subImageInColumnIndex, subImageInRowIndex);
    return [getColorValuesAtOrigin(origin.x, origin.y)];
}

function getColorValuesAtOrigin(originX, originY, width = subImageWidth, height = subImageHeight) {
    const colorValues = [];
    for(let j = originY; j < originY + height; j++) {
        for(let i = originX; i < originX + width; i++) {
            const currentPixelData = ctx.getImageData(i, j, 1, 1).data;
            colorValues.push(colorValueMapper(currentPixelData[0]));
            colorValues.push(colorValueMapper(currentPixelData[1]));
            colorValues.push(colorValueMapper(currentPixelData[2]));
        }
    }
    return colorValues;
}

function colorValueMapper(colour) {
    return (2 * colour / 255) - 1;
}

const neuralNetworkModule = (() => {
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

    function calculateResult(input, weights) {
        let activeLayerValues = input;
        weights.forEach(weightMatrix => {
            activeLayerValues = matrixMultiply(activeLayerValues, weightMatrix);
        })
        return activeLayerValues;
    }

    return {
        matrixMultiply,
        calculateResult,
    }
})();

function drawSubImage(colorValues, ctx, originX, originY, width = subImageWidth, height = subImageHeight) {
    const subImageData = ctx.createImageData(width, height);
    for(let currColorValueIndex = 0, imageDataIndex = 0;
        currColorValueIndex < colorValues.length;
        currColorValueIndex += 3, imageDataIndex += 4)
    {
        subImageData.data[imageDataIndex] = colorValues[currColorValueIndex];
        subImageData.data[imageDataIndex + 1] = colorValues[currColorValueIndex + 1];
        subImageData.data[imageDataIndex + 2] = colorValues[currColorValueIndex + 2];
        subImageData.data[imageDataIndex + 3] = 255;
    }
    ctx.putImageData(subImageData, originX, originY);
}

function valueColorMapper(colour) {
    if (colour > 1) return 255;
    if (colour < -1) return 0;
    return (colour + 1) * 255 / 2;
}

document.getElementById('reset-button').addEventListener('click', () => {
    const secondLayerNeurons = +prompt('Count of neurons on the second layer:');
    if (!secondLayerNeurons || isNaN(secondLayerNeurons) || secondLayerNeurons % 1 !== 0) return;
    const error = +prompt('Maximum error:');
    if (!error || isNaN(error)) return;

    const weights = setRandomWeights(secondLayerNeurons);
    learningParameters.currWeights = weights;
    learningParameters.maxError = error;
});

function setRandomWeights(secondLayerNeurons) {
    const firstWeights = [];
    for(let i = 0; i < pixelsInSubImage * valuesInAPixel; i++) {
        const newRow = [];
        for(let j = 0; j < secondLayerNeurons; j++) {
            newRow.push(Math.random() * (Math.random() > 0.5 ? 1 : -1));
        }
        firstWeights.push(newRow);
    }
    const secondWeights = transposingMatrix(firstWeights);
    return [firstWeights, secondWeights];
}

function transposingMatrix(matrix) {
    const transposed = [];
    for(let j = 0; j < matrix[0].length; j++) {
        transposed.push(new Array(matrix.length));
        for(let i = 0; i < matrix.length; i++) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

document.getElementById('next-learning-step-button').addEventListener('click', () => {
    nextLearningStep();
    alert('Done!');
});

function nextLearningStep() {
    //for(let i = 0; i < learningParameters.currWeights.length; i++) {
    for(let i = 0; i < learningParameters.parameters.length; i++) {
        tuneWeights(learningParameters.currWeights, learningParameters.maxError, learningParameters.parameters[i]);
    }
    compressAndDecompress(learningParameters);
    //}
}

function tuneWeights(weights, maxError, parameters) {
    if (!weights || !parameters || !parameters.currInput || !parameters.secondLayerValues || !parameters.currOutput) return;
    const X = parameters.currInput;
    let Y = parameters.secondLayerValues;
    let Xtick = parameters.currOutput;
    let diff = differenceMatrices(Xtick, X);
    let currError;
    do {
        let i = 0;
        tuneSecondWeights(weights, Y, diff);
        i++;
        tuneFirstWeights(weights, X, diff);
        normalizeByRow(weights[1]);
        normalizeByColumn(weights[0]);
        Y = neuralNetworkModule.matrixMultiply(X, weights[0]);
        Xtick = neuralNetworkModule.matrixMultiply(Y, weights[1]);
        diff = differenceMatrices(Xtick, X);
        currError = vectorSquare(diff[0]);
    }
    while(currError > maxError);
}

function normalizeByRow(weightMatrix) {
    weightMatrix.forEach((row, index) => normalizeRow(weightMatrix, index));
}

function normalizeRow(matrix, indexRow) {
    const modulus = Math.sqrt(matrix[indexRow].map(el => el * el).reduce((r, v) => r + v, 0));
    for(let j = 0; j < matrix[indexRow].length; j++) {
        matrix[indexRow][j] /= modulus;
    }
}

function normalizeByColumn(weightMatrix) {
    for(let j = 0; j < weightMatrix[0].length; j++) {
        let columnSum = 0;
        for(let i = 0; i < weightMatrix.length; i++) {
            columnSum += Math.pow(weightMatrix[i][j], 2);
        }
        const modulus = Math.sqrt(columnSum);
        for(let i = 0; i < weightMatrix.length; i++) {
            weightMatrix[i][j] /= modulus;
        }
    }
}
function differenceMatrices(A, B) {
    const result = [];
    const height = A.length;
    const width = A[0].length;
    for (let i = 0; i < height; i++) {
        result.push(new Array(width));
        for(let j = 0; j < width; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

function tuneFirstWeights(weights, A, diff) {
    const inputTransposed = transposingMatrix(A);
    const a = neuralNetworkModule.matrixMultiply(inputTransposed, diff);
    const b = neuralNetworkModule.matrixMultiply(a, transposingMatrix(weights[1]));
    const learningCoefficient = 1 / vectorSquare(transposingMatrix(inputTransposed)[0]);
    if(!isFinite(learningCoefficient)) return;
    multiplyMatrixByNumber(b, learningCoefficient);
    weights[0] = differenceMatrices(weights[0], b);
}

function tuneSecondWeights(weights, B, diff) {
    const secondLayerValuesTransposed = transposingMatrix(B);
    const a = neuralNetworkModule.matrixMultiply(secondLayerValuesTransposed, diff);
    const learningCoefficient = 1 / vectorSquare(transposingMatrix(secondLayerValuesTransposed)[0]);
    if(!isFinite(learningCoefficient)) return;
    multiplyMatrixByNumber(a, learningCoefficient);
    weights[1] = differenceMatrices(weights[1], a);
}

function vectorSquare(vector) {
    return vector.reduce((r, v) => r + v*v, 0);
}

function multiplyMatrixByNumber(matrix, number) {
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            matrix[i][j] *= number;
        }
    }
}

document.getElementById('next-n-learning-steps-button').addEventListener('click', () => {
    const steps = +prompt("Learning steps:");
    if (!steps || isNaN(steps) || steps <= 0 || steps % 1 !== 0) return;
    for(let iter = 0; iter < steps; iter++) {
        nextLearningStep();
    }
    alert('Done!');
});

const subImageWidth = 2;
const subImageHeight = 2;
const pixelsInSubImage = subImageWidth * subImageHeight;
const imageWidth = 256;
const imageHeight = 256;
const subImagesInOneRow = imageWidth / subImageWidth;
const subImagesInOneColumn = imageHeight / subImageHeight;
const valuesInAPixel = 3;
const learningParameters = {};

const canvas = document.getElementById("image-container");
const ctx = canvas.getContext("2d");
const imageInput = document.getElementById("image-input");
imageInput.addEventListener("change", function() {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
        const uploadedImage = new Image();
        uploadedImage.addEventListener("load", () => {
            ctx.drawImage(uploadedImage, 0, 0);
            ctx.createImageData(256, 256);
        });
        uploadedImage.src = reader.result;
    });
    reader.readAsDataURL(this.files[0]);
});

const compressedImageCtx = document.getElementById('compressed-image-container').getContext("2d");
