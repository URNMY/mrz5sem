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

    function normalizeByRow(weightMatrix) {
        weightMatrix.forEach((row, index) => normalizeRow(weightMatrix, index));
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

    function inverseMatrix(matrix) {
        const determinant = calculateDet(matrix);
        const transposed = transposingMatrix(matrix);
        const inverseHeight = matrix[0].length;
        const inverseWidth = matrix.length;
        for (let i = 0; i < inverseHeight; i++) {
            for (let j = 0; j < inverseWidth; j++) {
                if ((i + j) % 2 === 0) {
                    transposed[i][j] = -transposed[i][j];
                }
            }
        }
        return multiplyMatrixByNumber(transposed, 1 / determinant);
    }

    function calculateDet(matrix) {
        try {
            if (matrix.length !== matrix[0].length) {
                throw "Non-square matrix passed to the calculateDet function";
            }
            if (matrix.length === 1) {
                return matrix[0][0];
            }
            else {
                let determinant = 0;
                for (let column = 0; column < matrix[0].length; column++) {
                    const currMultiplier = matrix[0][column];
                    const sign = column % 2 === 0 ? 1 : -1;
                    determinant += calculateDet(ignoreIthRowAndJthColumn(matrix, 0, column)) * currMultiplier * sign;
                }
                return determinant;
            }
        }
        catch(err) {
            console.error(err);
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

    function areVectorsLinearlyIndependent(vectors) {
        for (let i = 0; i < vectors.length; i++) {
            const currFirstComparedVector = vectors[i];
            for (let j = i + 1; j < vectors.length; j++) {
                const currSecondComparedVector = vectors[j];
                const currRatio = currFirstComparedVector[0] / currSecondComparedVector[0];
                if (currFirstComparedVector.some((el, index) => el / currSecondComparedVector[index] !== currRatio)) return true;
            }
        }
        return false;
    }

    function getRandMatrix(height, width, zeroedMainDiagonal = true, bottomClamp = -1, topClamp = 1) {
        const matrix = new Array(height).fill().map(el => new Array(width));
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                if (zeroedMainDiagonal && i === j) {
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

    return {
        matrixMultiply,
        transposingMatrix,
        differenceMatrices,
        normalizeByRow,
        normalizeByColumn,
        multiplyMatrixByNumber,
        inverseMatrix,
        areVectorsLinearlyIndependent,
        getRandMatrix,
        applyFunctionToMatrixElements,
        calculateDet
    }
})();

const inputVectorSize = 900;

const canvasModule = (function() {
    const pixelsInDimension = Math.floor(Math.sqrt(inputVectorSize));
    const pixelSize = 8;
    const canvasSize = pixelSize * pixelsInDimension;
    const strCanvasSize = canvasSize.toString();
    const inputCanvas = document.getElementById('input');
    const outputCanvas = document.getElementById('output');
    inputCanvas.setAttribute('width', strCanvasSize);
    inputCanvas.setAttribute('height', strCanvasSize);
    outputCanvas.setAttribute('width', strCanvasSize);
    outputCanvas.setAttribute('height', strCanvasSize);

    const inputContext = inputCanvas.getContext('2d');
    const outputContext = outputCanvas.getContext('2d');

    function getVector(vector, isInput = true) {
        if (vector.length !== Math.pow(pixelsInDimension, 2)) return;
        vector.forEach((el, index) => drawPixel(index, el === 1, isInput));
    }

    function getPixelOrigin(index, size = pixelSize) {
        const pixelInColumnIndex = index % pixelsInDimension;
        const pixelInRowIndex = Math.floor(index / pixelsInDimension);
        return {
            x: pixelInColumnIndex * size,
            y: pixelInRowIndex * size
        };
    }

    function drawPixel(index, isWhite, isInput = true) {
        const ctx = (isInput ? inputContext : outputContext);
        const imageData = ctx.createImageData(pixelSize, pixelSize);
        for (let i = 0; i < Math.pow(pixelSize, 2); i++) {
            const colorValue = isWhite ? 255 : 0;
            imageData.data[i * 4] = colorValue;
            imageData.data[i * 4 + 1] = colorValue;
            imageData.data[i * 4 + 2] = colorValue;
            imageData.data[i * 4 + 3] = 255;
        }
        const origin = getPixelOrigin(index);
        ctx.putImageData(imageData, origin.x, origin.y);
    }

    return {
        getVector
    }
})();

const hammingNetworkModule = (function() {
    const activationFunction = Math.tanh;
    const inputElementsQuantity = inputVectorSize;
    const weightMatrixDimension = inputElementsQuantity;

    let vectorInANetwork;
    let weightMatrix;

    function init() {
        vectorInANetwork = [];
        weightMatrix = matrixModule.getRandMatrix(weightMatrixDimension, weightMatrixDimension);
    }

    function getNetworkOutput(inputVector) {
        const output = matrixModule.matrixMultiply(inputVector, weightMatrix);
        const outputWithActivationFunction = matrixModule.applyFunctionToMatrixElements(output, activationFunction);
        const outputWithReducer = matrixModule.applyFunctionToMatrixElements(outputWithActivationFunction, valueReducer);
        return outputWithReducer;
    }

    function addVectorToMemory(vector) {
        vectorInANetwork.push(vector);
    }

    function learningStep() {
        if (!vectorInANetwork.length) return;
        const X = matrixModule.transposingMatrix(vectorInANetwork);
        const Xtranspose = matrixModule.transposingMatrix(X);
        const middleTerm = matrixModule.inverseMatrix(matrixModule.matrixMultiply(Xtranspose, X));
        weightMatrix = matrixModule.matrixMultiply(matrixModule.matrixMultiply(X, middleTerm), Xtranspose);
        console.log(vectorInANetwork);
        console.log(weightMatrix);
    }

    function valueReducer(value) {
        return value >= 0 ? -1 : 1;
    }

    function checkVectorAlreadyInANetwork(vector) {
        return vectorInANetwork.some(rememberedVector => rememberedVector.every((el, index) => el === vector[index]));
    }

    return {
        init,
        getNetworkOutput,
        addVectorToMemory,
        learningStep,
        checkVectorAlreadyInANetwork
    }
})();

const fileReaderModule = (function() {
    let currV;

    document.getElementById('text-file-input').addEventListener('change', function() {
        try {
            const reader = new FileReader();
            reader.onload = () => {
                currV = convertStringToVector(reader.result);
                if (currV.length !== inputVectorSize) {
                    throw `Invalid input vector size: wanted ${inputVectorSize}, got ${currV.length}`;
                }
                canvasModule.getVector(currV);
            }
            reader.readAsText(this.files[0]);
        }
        catch(err) {
            console.error(err);
        }
    });

    function getCurrV() {
        return currV;
    }

    function convertStringToVector(string) {
        return string.split('\n').join(' ').split(' ').filter(el => !!el).map(Number);
    }

    return {
        getCurrV
    }
})();

document.getElementById('convert').addEventListener('click', () => {
    canvasModule.getVector(hammingNetworkModule.getNetworkOutput([fileReaderModule.getCurrV()])[0], false);
});

document.getElementById('save-for-learning').addEventListener('click', () => {
    const currV = fileReaderModule.getCurrV();
    if (!currV) return;
    if (hammingNetworkModule.checkVectorAlreadyInANetwork(currV)) {
        alert("this vector is already saved for learning");
        return;
    }
    hammingNetworkModule.addVectorToMemory(currV);
    alert('saved!');
});

document.getElementById('learning-step').addEventListener('click', () => {
    hammingNetworkModule.learningStep();
    alert('finished!');
});

document.getElementById('reset').addEventListener('click', () => {
    hammingNetworkModule.init();
});

hammingNetworkModule.init();
