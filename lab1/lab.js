document.getElementById('compress-button').addEventListener('click', () => {
    try {
        de_Compress(learningParameters);
    }
    catch(err) {
        alert('DE_COMPRESSING ERROR!');
        console.error(err);
    }
});

function de_Compress(learningParameters) {
    learningParameters.parameters = [];
    const weights = learningParameters.currentWeights;
    for(let imageSquareInColumnIndex = 0; imageSquareInColumnIndex < innerPixelsInOneColumn; imageSquareInColumnIndex++) {
        for(let imageSquareInRowIndex = 0; imageSquareInRowIndex < innerPixelsInOneRow; imageSquareInRowIndex++) {
            const innerPicture = getInnerPicture(imageSquareInColumnIndex, imageSquareInRowIndex);
            const currInput = getNeuralNetworkImageSquareInput(imageSquareInColumnIndex, imageSquareInRowIndex);
            const secondLayerValues = LinearRecirculationNetwork.multiplyMatrices(currInput, weights[0]);
            const currOutput = LinearRecirculationNetwork.multiplyMatrices(secondLayerValues, weights[1]);
            drawImageSquare(
                currOutput[0].map(ColourValueMapper((colour + 1) * 255 / 2)),
                compressedImageCtx,
                innerPicture.x,
                innerPicture.y
            )
            learningParameters.parameters.push({
                currInput,
                secondLayerValues,
                currOutput,
            });
        }
    }
}

function getInnerPicture(imageSquareInColumnIndex, imageSquareInRowIndex, width = innerBlockWidth, height = innerBlockHeight) {
    return {
        x: imageSquareInRowIndex * width,
        y: imageSquareInColumnIndex * height
    };
}

function getNeuralNetworkImageSquareInput(imageSquareInColumnIndex, imageSquareInRowIndex) {
    const innerPicture = getInnerPicture(imageSquareInColumnIndex, imageSquareInRowIndex);
    return [getColourValuesAtinnerPicture(innerPicture.x, innerPicture.y)];
}

function getColourValuesAtinnerPicture(innerPictureX, innerPictureY, width = innerBlockWidth, height = innerBlockHeight) {
    const colourValues = [];
    for(let j = innerPictureY; j < innerPictureY + height; j++) {
        for(let i = innerPictureX; i < innerPictureX + width; i++) {
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

    return {
        multiplyMatrices,
    }
})();

function drawImageSquare(colourValues, ctx, innerPictureX, innerPictureY, width = innerBlockWidth, height = innerBlockHeight) {
    const imageSquareData = ctx.createImageData(width, height);
    for(let currColourValueIndex = 0, imageDataIndex = 0;
        currColourValueIndex < colourValues.length;
        currColourValueIndex += 3, imageDataIndex += 4)
    {
        imageSquareData.data[imageDataIndex] = colourValues[currColourValueIndex];
        imageSquareData.data[imageDataIndex + 1] = colourValues[currColourValueIndex + 1];
        imageSquareData.data[imageDataIndex + 2] = colourValues[currColourValueIndex + 2];
        imageSquareData.data[imageDataIndex + 3] = 255;
    }
    ctx.putImageData(imageSquareData, innerPictureX, innerPictureY);
}

document.getElementById('reset-button').addEventListener('click', () => {
    const secondLayerNeurons = +prompt('Count of neurons on the second layer:');
    if (!secondLayerNeurons || isNaN(secondLayerNeurons) || secondLayerNeurons % 1 !== 0) return;
    const errorLimit = +prompt('Error limit:');
    if (!errorLimit || isNaN(errorLimit)) return;

    const weights = setRandomWeights(secondLayerNeurons);
    learningParameters.currentWeights = weights;
    learningParameters.maxErr = errorLimit;
});

function setRandomWeights(secondLayerNeurons) {
    const firstWeights = [];
    for(let i = 0; i < pixelsInImageSquare * valuesInAPixel; i++) {
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
    const transposing = [];
    for(let j = 0; j < matrix[0].length; j++) {
        transposing.push(new Array(matrix.length));
        for(let i = 0; i < matrix.length; i++) {
            transposing[j][i] = matrix[i][j];
        }
    }
    return transposing;
}

document.getElementById('learning-step-button').addEventListener('click', () => {
    for(let i = 0; i < learningParameters.parameters.length; i++) {
        tuneWeights(learningParameters.currentWeights, learningParameters.maxErr, learningParameters.parameters[i]);
    }
    de_Compress(learningParameters);
    alert('Done!');
});

function tuneWeights(weights, maxErr, parameters) {
    if (!weights || !parameters || !parameters.currInput || !parameters.secondLayerValues || !parameters.currOutput) return;
    const X = parameters.currInput;
    let Y = parameters.secondLayerValues;
    let Xoverlined = parameters.currOutput;
    let diff = differenceMatrices(Xoverlined, X);
    let currError;
    do {
        let i = 0;
        const secondLayerValuesTransposing = transposingMatrix(Y);
        let a = LinearRecirculationNetwork.multiplyMatrices(secondLayerValuesTransposing, diff);
        let learningCoefficient = 1 / transposingMatrix(secondLayerValuesTransposing)[0].reduce((row, value) => row + value*value, 0);
        if(!isFinite(learningCoefficient)) return;
        multiplyMatrixByNumber(a, learningCoefficient);
        weights[1] = differenceMatrices(weights[1], a);
        i++;
        const inputTransposing = transposingMatrix(X);
        a = LinearRecirculationNetwork.multiplyMatrices(inputTransposing, diff);
        const b = LinearRecirculationNetwork.multiplyMatrices(a, transposingMatrix(weights[1]));
        learningCoefficient = 1 / transposingMatrix(inputTransposing)[0].reduce((row, value) => row + value*value, 0)
        if(!isFinite(learningCoefficient)) return;
        multiplyMatrixByNumber(b, learningCoefficient);
        weights[0] = differenceMatrices(weights[0], b);
        normalizeByRow(weights[1]);
        normalizeByColumn(weights[0]);
        Y = LinearRecirculationNetwork.multiplyMatrices(X, weights[0]);
        Xoverlined = LinearRecirculationNetwork.multiplyMatrices(Y, weights[1]);
        diff = differenceMatrices(Xoverlined, X);
        currError = diff[0].reduce((row, value) => row + value*value, 0);
    }
    while(currError > maxErr);
}

function normalizeByRow(weightMatrix) {
    weightMatrix.forEach((row, index) => normalizeRow(weightMatrix, index));
}

function normalizeRow(matrix, indexRow) {
    const modulus = Math.sqrt(matrix[indexRow].map(el => el * el).reduce((row, value) => row + value, 0));
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
        for(let i = 0; i < learningParameters.parameters.length; i++) {
            tuneWeights(learningParameters.currentWeights, learningParameters.maxErr, learningParameters.parameters[i]);
        }
        de_Compress(learningParameters);
    }
    alert('Done!');
});

const innerBlockWidth = 2;
const innerBlockHeight = 2;
const pixelsInImageSquare = innerBlockWidth * innerBlockHeight;
const imageWidth = 256;
const imageHeight = 256;
const innerPixelsInOneRow = imageWidth / innerBlockWidth;
const innerPixelsInOneColumn = imageHeight / innerBlockHeight;
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
            ctx.createImageData(imageWidth, imageHeight);
        });
        uploadedImage.src = reader.result;
    });
    reader.readAsDataURL(this.files[0]);
});

const compressedImageCtx = document.getElementById('compressed-image-container').getContext("2d");
