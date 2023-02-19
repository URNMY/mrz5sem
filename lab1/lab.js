const square = 4;
const innerBlockWidth = Math.sqrt(square);
const innerBlockHeight = Math.sqrt(square);
const pixelsInImageSquare = innerBlockWidth * innerBlockHeight;
const imageWidth = 256;
const imageHeight = 256;
const innerPixelsInOneRow = imageWidth / innerBlockWidth;
const innerPixelsInOneColumn = imageHeight / innerBlockHeight;
const valuesInAPixel = 3;
const learningParameters = {};
const compressedImageCtx = document.getElementById('compressed-image-container').getContext("2d");
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

document.getElementById('button-compress').addEventListener('click', () => {
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
    let imageSquareInColumnIndex = 0;
    while(imageSquareInColumnIndex < innerPixelsInOneColumn){
        let imageSquareInRowIndex = 0;
        while(imageSquareInRowIndex < innerPixelsInOneRow) {
            const innerPicture = getInnerPicture(imageSquareInColumnIndex, imageSquareInRowIndex);
            const currInput = getNeuralNetworkImageSquareInput(imageSquareInColumnIndex, imageSquareInRowIndex);
            const secondLayerValues = multiplyMatrices(currInput, weights[0]);
            const currOutput = multiplyMatrices(secondLayerValues, weights[1]);
            drawImageSquare(
                currOutput[0].map(((2 * (colour + 1) * 255 / 2) / 255) - 1),
                compressedImageCtx,
                innerPicture.x,
                innerPicture.y
            )
            learningParameters.parameters.push({
                currInput,
                secondLayerValues,
                currOutput,
            });
            imageSquareInRowIndex++;
        }
        imageSquareInColumnIndex++;
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
    return [getColourValuesATinyPicture(innerPicture.x, innerPicture.y)];
}

function getColourValuesATinyPicture(innerPictureX, innerPictureY, width = innerBlockWidth, height = innerBlockHeight) {
    const colourValues = [];
    let j = innerPictureY;
    while (j < innerPictureY + height) {
        let i = innerPictureX;
        while (i < innerPictureX + width) {
            const currentPixelData = ctx.getImageData(i, j, 1, 1).data;
            for (let k = 0; k < square - 1; k++) {
                colourValues.push((2 * currentPixelData[k] / 255) - 1);
            }
            i++;
        }
        j++;
    }
    return colourValues;
}
    function multiplyMatrices(A, B)
    {
        const rowsA = A.length, rowsB = B.length, columnsB = B[0].length, C = [];
        let i = 0;
        while(i < rowsA) {
                C[i] = []
                i++;
        }
        let k = 0;
        while(k < columnsB) {
            let i = 0;
            while(i < rowsA){
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

function drawImageSquare(colourValues, ctx, innerPictureX, innerPictureY, width = innerBlockWidth, height = innerBlockHeight) {
    const imageSquareData = ctx.createImageData(width, height), maxValue = 255;
    let currColourValueIndex = 0, imageDataIndex = 0;
    while (currColourValueIndex < colourValues.length){
        for (let i = 0; i < square; i++) {
            if (i !== square - 1){
                imageSquareData.data[imageDataIndex + i] = colourValues[currColourValueIndex + i];
            }
            else {
                imageSquareData.data[imageDataIndex + i] = maxValue;
            }
        }
        currColourValueIndex += 3;
        imageDataIndex += 4;
    }
    ctx.putImageData(imageSquareData, innerPictureX, innerPictureY);
}

document.getElementById('button-reset').addEventListener('click', () => {
    const secondLayerNeurons = +prompt('Count of neurons on the second layer:');
    if (!secondLayerNeurons || isNaN(secondLayerNeurons) || secondLayerNeurons % 1 !== 0) return;
    const errorLimit = +prompt('Error limit:');
    if (!errorLimit || isNaN(errorLimit)) return;
    learningParameters.currentWeights = setRandomWeights(secondLayerNeurons);
    learningParameters.maxErr = errorLimit;
});

function setRandomWeights(secondLayerNeurons) {
    const firstWeights = [];
    let i = 0;
    while (i < pixelsInImageSquare * valuesInAPixel) {
        const newRow = [];
        let j = 0;
        while (j < secondLayerNeurons) {
            newRow.push(Math.random() * (Math.random() > 0.5 ? 1 : -1));
            j++;
        }
        firstWeights.push(newRow);
        i++;
    }
    const secondWeights = transposingMatrix(firstWeights);
    return [firstWeights, secondWeights];
}

function transposingMatrix(matrix) {
    const transposing = [];
    let j = 0;
    while(j < matrix[0].length) {
        transposing.push(new Array(matrix.length));
        let i = 0;
        while(i < matrix.length){
            transposing[j][i] = matrix[i][j];
            i++;
        }
        j++;
    }
    return transposing;
}

document.getElementById('button-learning-step').addEventListener('click', () => {
    let i = 0;
    while (i < learningParameters.parameters.length) {
        tuneWeights(learningParameters.currentWeights, learningParameters.maxErr, learningParameters.parameters[i]);
        i++;
    }
    de_Compress(learningParameters);
    alert('Done!');
});

function tuneWeights(weights, maxErr, parameters) {
    if (!weights || !parameters || !parameters.currInput || !parameters.secondLayerValues || !parameters.currOutput) return;
    const X = parameters.currInput;
    let Y = parameters.secondLayerValues, XOverlined = parameters.currOutput,
        diff = differenceMatrices(XOverlined, X), currError = maxErr + 1;
    while(currError > maxErr) {
        const secondLayerValuesTransposing = transposingMatrix(Y);
        const inputTransposing = transposingMatrix(X);
        let a = multiplyMatrices(secondLayerValuesTransposing, diff);
        let learningCoefficient = 1 / transposingMatrix(secondLayerValuesTransposing)[0].reduce((row, value) => row + value*value, 0);
        if(!isFinite(learningCoefficient)) return;
        multiplyMatrixByNumber(a, learningCoefficient);
        weights[1] = differenceMatrices(weights[1], a);
        a = multiplyMatrices(inputTransposing, diff);
        const b = multiplyMatrices(a, transposingMatrix(weights[1]));
        learningCoefficient = 1 / transposingMatrix(inputTransposing)[0].reduce((row, value) => row + value*value, 0)
        if(!isFinite(learningCoefficient)) return;
        multiplyMatrixByNumber(b, learningCoefficient);
        weights[0] = differenceMatrices(weights[0], b);
        weights[1].forEach((row, index) => normalizeRow(weights[1], index));
        normalizeByColumn(weights[0]);
        Y = multiplyMatrices(X, weights[0]);
        XOverlined = multiplyMatrices(Y, weights[1]);
        diff = differenceMatrices(XOverlined, X);
        currError = diff[0].reduce((row, value) => row + value*value, 0);
    }
}

function normalizeRow(matrix, indexRow) {
    const modulus = Math.sqrt(matrix[indexRow].map(el => el * el).reduce((row, value) => row + value, 0));
    let i = 0;
    while(i < matrix[indexRow].length) {
        matrix[indexRow][i] /= modulus;
        i++;
    }
}

function normalizeByColumn(weightMatrix) {
    let j = 0;
    while(j < weightMatrix[0].length) {
        let columnSum = 0, i = 0;
        while(i < weightMatrix.length) {
            columnSum += Math.pow(weightMatrix[i][j], 2);
            i++;
        }
        const modulus = Math.sqrt(columnSum);
        let k = 0;
        while(k < weightMatrix.length) {
            weightMatrix[k][j] /= modulus;
            k++;
        }
        j++;
    }
}
function differenceMatrices(A, B) {
    const result = [], height = A.length, width = A[0].length;
    let i = 0;
    while(i < height) {
        result.push(new Array(width));
        let j = 0;
        while(j < width) {
            result[i][j] = A[i][j] - B[i][j];
            j++;
        }
        i++;
    }
    return result;
}

function multiplyMatrixByNumber(matrix, number) {
    let i = 0;
    while(i < matrix.length) {
        let j = 0;
        while(j < matrix[0].length){
            matrix[i][j] *= number;
            j++;
        }
        i++;
    }
}

document.getElementById('button-learning-steps').addEventListener('click', () => {
    const steps = +prompt("Learning steps:");
    if (!steps || isNaN(steps) || steps <= 0 || steps % 1 !== 0) return;
    let it = 0;
    while (it < steps) {
        let i = 0;
        while(i < learningParameters.parameters.length) {
            tuneWeights(learningParameters.currentWeights, learningParameters.maxErr, learningParameters.parameters[i]);
            i++;
        }
        de_Compress(learningParameters);
        it++;
    }
    alert('Done!');
});


