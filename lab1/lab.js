document.getElementById('compress-button').addEventListener('click', () => {
    try {
        de_Compress(learningParameters);
    }
    catch(err) {
        alert('DE_COMPRESSING ERROR!');
        console.error(err);
        return;
    }
});

function de_Compress(learningParameters) {
    learningParameters.parameters = [];
    const weights = learningParameters.currentWeights;
    for(let subImageInColumnIndex = 0; subImageInColumnIndex < subImagesInOneColumn; subImageInColumnIndex++) {
        for(let subImageInRowIndex = 0; subImageInRowIndex < subImagesInOneRow; subImageInRowIndex++) {
            const origin = getOrigin(subImageInColumnIndex, subImageInRowIndex);
            const currInput = getNeuralNetworkSubImageInput(subImageInColumnIndex, subImageInRowIndex);
            const secondLayerValues = LinearRecirculationNetwork.multiplyMatrices(currInput, weights[0]);
            const currOutput = LinearRecirculationNetwork.multiplyMatrices(secondLayerValues, weights[1]);
            drawSubImage(
                currOutput[0].map(refactorColourPixels),
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
    return [getColourValuesAtOrigin(origin.x, origin.y)];
}

function getColourValuesAtOrigin(originX, originY, width = subImageWidth, height = subImageHeight) {
    const colourValues = [];
    for(let j = originY; j < originY + height; j++) {
        for(let i = originX; i < originX + width; i++) {
            const currentPixelData = ctx.getImageData(i, j, 1, 1).data;
            colourValues.push(ColourValueMapper(currentPixelData[0]));
            colourValues.push(ColourValueMapper(currentPixelData[1]));
            colourValues.push(ColourValueMapper(currentPixelData[2]));
        }
    }
    return colourValues;
}

function ColourValueMapper(colour) {
    return (2 * colour / 255) - 1;
}

const LinearRecirculationNetwork = (() => {
    function multiplyMatrices(A, B)
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
            activeLayerValues = multiplyMatrices(activeLayerValues, weightMatrix);
        })
        return activeLayerValues;
    }

    return {
        multiplyMatrices,
        calculateResult,
    }
})();

function drawSubImage(colourValues, ctx, originX, originY, width = subImageWidth, height = subImageHeight) {
    const subImageData = ctx.createImageData(width, height);
    for(let currColourValueIndex = 0, imageDataIndex = 0;
        currColourValueIndex < colourValues.length;
        currColourValueIndex += 3, imageDataIndex += 4)
    {
        subImageData.data[imageDataIndex] = colourValues[currColourValueIndex];
        subImageData.data[imageDataIndex + 1] = colourValues[currColourValueIndex + 1];
        subImageData.data[imageDataIndex + 2] = colourValues[currColourValueIndex + 2];
        subImageData.data[imageDataIndex + 3] = 255;
    }
    ctx.putImageData(subImageData, originX, originY);
}

function refactorColourPixels(colour) {
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
    learningParameters.currentWeights = weights;
    learningParameters.maxErr = error;
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

document.getElementById('learning-step-button').addEventListener('click', () => {
    nextLearningStep();
    alert('Done!');
});

function nextLearningStep() {
    for(let i = 0; i < learningParameters.parameters.length; i++) {
        tuneWeights(learningParameters.currentWeights, learningParameters.maxErr, learningParameters.parameters[i]);
    }
    de_Compress(learningParameters);
}

function tuneWeights(weights, maxErr, parameters) {
    if (!weights || !parameters || !parameters.currInput || !parameters.secondLayerValues || !parameters.currOutput) return;
    const X = parameters.currInput;
    let Y = parameters.secondLayerValues;
    let Xoverlined = parameters.currOutput;
    let diff = differenceMatrices(Xoverlined, X);
    let currError;
    do {
        let i = 0;
        tuneSecondWeights(weights, Y, diff);
        i++;
        tuneFirstWeights(weights, X, diff);
        normalizeByRow(weights[1]);
        normalizeByColumn(weights[0]);
        Y = LinearRecirculationNetwork.multiplyMatrices(X, weights[0]);
        Xoverlined = LinearRecirculationNetwork.multiplyMatrices(Y, weights[1]);
        diff = differenceMatrices(Xoverlined, X);
        currError = vectSquare(diff[0]);
    }
    while(currError > maxErr);
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
    const a = LinearRecirculationNetwork.multiplyMatrices(inputTransposed, diff);
    const b = LinearRecirculationNetwork.multiplyMatrices(a, transposingMatrix(weights[1]));
    const learningCoefficient = 1 / vectSquare(transposingMatrix(inputTransposed)[0]);
    if(!isFinite(learningCoefficient)) return;
    multiplyMatrixByNumber(b, learningCoefficient);
    weights[0] = differenceMatrices(weights[0], b);
}

function tuneSecondWeights(weights, B, diff) {
    const secondLayerValuesTransposed = transposingMatrix(B);
    const a = LinearRecirculationNetwork.multiplyMatrices(secondLayerValuesTransposed, diff);
    const learningCoefficient = 1 / vectSquare(transposingMatrix(secondLayerValuesTransposed)[0]);
    if(!isFinite(learningCoefficient)) return;
    multiplyMatrixByNumber(a, learningCoefficient);
    weights[1] = differenceMatrices(weights[1], a);
}

function vectSquare(vect) {
    return vect.reduce((r, v) => r + v*v, 0);
}

function multiplyMatrixByNumber(matrix, number) {
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            matrix[i][j] *= number;
        }
    }
}

document.getElementById('learning-steps-button').addEventListener('click', () => {
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
