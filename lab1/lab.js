const square = 4,
     innerBlockWidth = Math.sqrt(square),
    innerBlockHeight = Math.sqrt(square),
     pixelsInImageSquare = innerBlockWidth * innerBlockHeight,
    innerPixelsInOneRow = imageWidth / innerBlockWidth,
    innerPixelsInOneColumn = imageHeight / innerBlockHeight,
    valuesInAPixel = 3,
    learningParameters = {},
    imageWidth = 256,
    imageHeight = 256,
    maxValue = 255,
    threshold = 0.5,
    concisedImageContext = document.getElementById('concised-image-container').getContext("2d"),
    canvas = document.getElementById("image-container"),
    context = canvas.getContext("2d"),
    imageInput = document.getElementById("area-image-input");

imageInput.addEventListener("change", function() {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
        const uploadImage = new Image();
        uploadImage.addEventListener("load", () => {
            context.drawImage(uploadImage, 0, 0);
            context.createImageData(imageWidth, imageHeight);
        });
        uploadImage.src = reader.result;
    });
    reader.readAsDataURL(this.files[0]);
});

document.getElementById('button-compress').addEventListener('click', () => {
        de_Compress(learningParameters);
});

function de_Compress(learningParameters) {
    learningParameters.parameters = [];
    const weights = learningParameters.currentWeights;
    let imageSquareInColumnIndex = 0;
    while(imageSquareInColumnIndex < innerPixelsInOneColumn){
        let imageSquareInRowIndex = 0;
        while(imageSquareInRowIndex < innerPixelsInOneRow) {
            let innerPicture.x = imageSquareInRowIndex * innerBlockWidth,
                  innerPicture.y = imageSquareInColumnIndex * innerBlockHeight,
                  currentInput = [getColourValuesATinyPicture(innerPicture.x, innerPicture.y)],
                  secondLayerValues = multiplyMatrices(currentInput, weights[0]),
                  currOutput = multiplyMatrices(secondLayerValues, weights[1]);
            drawImageSquare(
                currOutput[0].map(((2 * (++colour) * maxValue / 2) / maxValue) - 1),
                concisedImageContext,
                innerPicture.x,
                innerPicture.y
            )
            learningParameters.parameters.push({
                currentInput,
                secondLayerValues,
                currOutput,
            });
            imageSquareInRowIndex++;
        }
        imageSquareInColumnIndex++;
    }
}

function getColourValuesATinyPicture(innerPictureX, innerPictureY) {
    const colourValues = [];
    let j = innerPictureY;
    while (j < innerPictureY + innerBlockHeight) {
        let i = innerPictureX;
        while (i < innerPictureX + innerBlockWidth) {
            const currentPixelData = context.getImageData(i, j, 1, 1).data;
            for (let k = 0; k < square - 1; k++) {
                colourValues.push((2 * currentPixelData[k] / maxValue) - 1);
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

function drawImageSquare(colourValues, context, innerPictureX, innerPictureY) {
    const imageSquareData = context.createImageData(innerBlockWidth, innerBlockHeight);
    let colourValueIndex = 0, imageDataIndex = 0;
    while (colourValueIndex < colourValues.length){
        for (let i = 0; i < square; i++) {
            if (i !== square - 1){
                imageSquareData.data[imageDataIndex + i] = colourValues[colourValueIndex + i];
            }
            else {
                imageSquareData.data[imageDataIndex + i] = maxValue;
            }
        }
        colourValueIndex = colourValueIndex + 3;
        imageDataIndex = imageDataIndex + 4;
    }
    context.putImageData(imageSquareData, innerPictureX, innerPictureY);
}

document.getElementById('button-reset').addEventListener('click', () => {
    const secondLayerNeurons = +prompt('Count of neurons on the second layer:'),
        errorLimit = +prompt('Error limit:'),
        firstWeights = [];
    let i = 0;
    while (i < pixelsInImageSquare * valuesInAPixel) {
        const newRow = [];
        let j = 0;
        while (j < secondLayerNeurons) {
            newRow.push(Math.random() * (Math.random() > threshold ? 1 : -1));
            j++;
        }
        firstWeights.push(newRow);
        i++;
    }
    const secondWeights = transposingMatrix(firstWeights);
    learningParameters.currentWeights = [firstWeights, secondWeights];
    learningParameters.limError = errorLimit;
});

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
        tuningWeights(learningParameters.currentWeights, learningParameters.limError, learningParameters.parameters[i]);
        i++;
    }
    de_Compress(learningParameters);
    alert('Done!');
});

function tuningWeights(weights, limError, parameters) {
    const inputLayerX = parameters.currentInput, next = 1, previous = 0;
    let secondLayerY = parameters.secondLayerValues, XOverlined = parameters.currOutput,
        diff = differenceMatrices(XOverlined, inputLayerX), currError = limError + 1;
    while(currError > limError) {
        const secondLayerValuesTransposing = transposingMatrix(secondLayerY),
              inputTransposing = transposingMatrix(inputLayerX);
        let a = multiplyMatrices(secondLayerValuesTransposing, diff),
            learningCoefficient = Math.pow(transposingMatrix(secondLayerValuesTransposing)[0].reduce((row, value) => row + Math.pow(value,2), 0),-1);
        multiplyMatrixByNumber(a, learningCoefficient);
        a = multiplyMatrices(inputTransposing, diff);
        weights[next] = differenceMatrices(weights[next], a);
        const b = multiplyMatrices(a, transposingMatrix(weights[next]));
        learningCoefficient = Math.pow(transposingMatrix(inputTransposing)[0].reduce((row, value) => row + Math.pow(value,2), 0),-1);
        multiplyMatrixByNumber(b, learningCoefficient);
        weights[previous] = differenceMatrices(weights[previous], b);
        weights[next].forEach((row, index) => normalizeRow(weights[next], index));
        let j = 0;
        while(j < weights[previous][previous].length) {
            let columnSum = 0, i = 0;
            while(i < weights[previous].length) {
                columnSum += Math.pow(weights[previous][i][j], 2);
                i++;
            }
            const modulus = Math.sqrt(columnSum);
            let k = 0;
            while(k < weights[previous].length) {
                weights[previous][k][j] = weights[previous][k][j] / modulus;
                k++;
            }
            j++;
        }
        secondLayerY = multiplyMatrices(inputLayerX, weights[0]);
        XOverlined = multiplyMatrices(secondLayerY, weights[1]);
        diff = differenceMatrices(XOverlined, inputLayerX);
        currError = diff[0].reduce((row, value) => row + Math.pow(value,2), 0);
    }
}

function normalizeRow(matrix, indexRow) {
    const modulus = Math.sqrt(matrix[indexRow].map(el => el * el).reduce((row, value) => row + value, 0));
    let i = 0;
    while(i < matrix[indexRow].length) {
        matrix[indexRow][i] = matrix[indexRow][i] / modulus;
        i++;
    }
}
function differenceMatrices(A, B) {
    const result = [];
    let i = 0;
    while(i < A.length) {
        result.push(new Array(A[0].length));
        let j = 0;
        while(j < A[0].length) {
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
            matrix[i][j] = matrix[i][j] * number;
            j++;
        }
        i++;
    }
}

document.getElementById('button-learning-steps').addEventListener('click', () => {
    const steps = +prompt("Learning steps:");
    let it = 0;
    while (it < steps) {
        let i = 0;
        while(i < learningParameters.parameters.length) {
            tuningWeights(learningParameters.currentWeights, learningParameters.limError, learningParameters.parameters[i]);
            i++;
        }
        de_Compress(learningParameters);
        it++;
    }
    alert('Done!');
});


