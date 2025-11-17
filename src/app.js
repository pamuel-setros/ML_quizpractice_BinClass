// ============================================================================
// Data Generation
// ============================================================================

class DataGenerator {
    static generate(numPoints, clusterTightness, ringWidth, mixingPercent) {
        const data = [];
        const ringInner = 0.25;
        const ringOuter = ringInner + (ringWidth * 0.15);
        
        // Distribute points: main clusters + mixed points
        const totalBluePoints = Math.floor(numPoints / 2);
        const totalRedPoints = Math.floor(numPoints / 2);

        // Generate blue points (center cluster) - some pure, some mixed
        const mixedBlueCount = Math.floor(totalBluePoints * mixingPercent / 100);
        const pureBlueCount = totalBluePoints - mixedBlueCount;
        
        // Pure blue points in center
        for (let i = 0; i < pureBlueCount; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = Math.abs(Math.random() * clusterTightness);
            const x = 0.5 + (radius * Math.cos(angle)) * 0.15;
            const y = 0.5 + (radius * Math.sin(angle)) * 0.15;
            data.push({ x, y, label: 0, color: 'blue' });
        }

        // Generate red points (ring) - some pure, some mixed
        const mixedRedCount = Math.floor(totalRedPoints * mixingPercent / 100);
        const pureRedCount = totalRedPoints - mixedRedCount;
        
        // Pure red points in ring
        for (let i = 0; i < pureRedCount; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = ringInner + Math.random() * (ringOuter - ringInner);
            const x = 0.5 + radius * Math.cos(angle);
            const y = 0.5 + radius * Math.sin(angle);
            data.push({ x, y, label: 1, color: 'red' });
        }

        // Mixed blue points: place them in the ring region
        for (let i = 0; i < mixedBlueCount; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = ringInner + Math.random() * (ringOuter - ringInner);
            const x = 0.5 + radius * Math.cos(angle);
            const y = 0.5 + radius * Math.sin(angle);
            data.push({ x, y, label: 0, color: 'blue' });
        }

        // Mixed red points: place them in the center region
        for (let i = 0; i < mixedRedCount; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = Math.abs(Math.random() * clusterTightness);
            const x = 0.5 + (radius * Math.cos(angle)) * 0.15;
            const y = 0.5 + (radius * Math.sin(angle)) * 0.15;
            data.push({ x, y, label: 1, color: 'red' });
        }

        return data;
    }
}

// ============================================================================
// Neural Network Implementation
// ============================================================================

class NeuralNetwork {
    constructor(inputSize, hiddenLayers, neuronsPerLayer, outputSize = 1) {
        this.layers = [];
        
        // Build network architecture
        const layerSizes = [inputSize];
        for (let i = 0; i < hiddenLayers; i++) {
            layerSizes.push(neuronsPerLayer);
        }
        layerSizes.push(outputSize);

        // Initialize layers with random weights and biases using Xavier initialization
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const inSize = layerSizes[i];
            const outSize = layerSizes[i + 1];
            const limit = Math.sqrt(6 / (inSize + outSize));
            
            const layer = {
                weights: this.randomMatrixXavier(inSize, outSize, limit),
                biases: new Array(outSize).fill(0)
            };
            this.layers.push(layer);
        }

        this.architecture = layerSizes;
        this.loss = 0;
    }

    randomMatrixXavier(rows, cols, limit) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() * 2 - 1) * limit;
            }
        }
        return matrix;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    relu(x) {
        return Math.max(0, x);
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    forward(input) {
        let activation = input;
        const activations = [activation];

        // Forward pass through all layers
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const output = [];
            
            for (let j = 0; j < layer.weights[0].length; j++) {
                let sum = layer.biases[j];
                for (let k = 0; k < activation.length; k++) {
                    sum += activation[k] * layer.weights[k][j];
                }
                
                // Use ReLU for hidden layers, sigmoid for output
                if (i < this.layers.length - 1) {
                    output[j] = this.relu(sum);
                } else {
                    output[j] = this.sigmoid(sum);
                }
            }
            
            activation = output;
            activations.push(activation);
        }

        return { output: activation, activations };
    }

    backward(input, target, learningRate) {
        const forward = this.forward(input);
        const activations = forward.activations;
        const prediction = forward.output[0];

        // Calculate output layer error
        const error = prediction - target;
        this.loss = error * error; // MSE loss

        // Initialize deltas array - one delta per layer output
        const deltas = [];
        
        // Output layer delta
        deltas[this.layers.length] = [error * prediction * (1 - prediction)];

        // Backpropagate through hidden layers
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            const nextDeltas = deltas[i + 1];
            const currentLayerDeltas = [];
            
            // For each neuron in current layer
            for (let j = 0; j < activations[i].length; j++) {
                let sum = 0;
                
                // Sum contributions from next layer
                for (let k = 0; k < nextDeltas.length; k++) {
                    sum += nextDeltas[k] * layer.weights[j][k];
                }
                
                // Apply activation derivative
                if (i > 0) {
                    // Hidden layer - use ReLU derivative
                    currentLayerDeltas.push(sum * this.reluDerivative(activations[i][j]));
                } else {
                    // Input layer - no derivative needed, but store for reference
                    currentLayerDeltas.push(sum);
                }
            }
            
            deltas[i] = currentLayerDeltas;
        }

        // Update weights and biases
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const nextDeltas = deltas[i + 1];
            const currentActivations = activations[i];
            
            // Update weights
            for (let j = 0; j < layer.weights.length; j++) {
                for (let k = 0; k < layer.weights[j].length; k++) {
                    const delta = nextDeltas[k];
                    layer.weights[j][k] -= learningRate * delta * currentActivations[j];
                }
            }
            
            // Update biases
            for (let k = 0; k < layer.biases.length; k++) {
                const delta = nextDeltas[k];
                layer.biases[k] -= learningRate * delta;
            }
        }
    }

    predict(input) {
        return this.forward(input).output[0];
    }
}

// ============================================================================
// Visualization and Main Application
// ============================================================================

class App {
    constructor() {
        this.canvas = document.getElementById('mainCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.networkCanvas = document.getElementById('networkCanvas');
        this.networkCtx = this.networkCanvas.getContext('2d');

        this.data = [];
        this.network = null;
        this.isTraining = false;
        this.currentEpoch = 0;
        this.totalEpochs = 100;

        this.setupEventListeners();
        this.generateInitialData();
        
        // Start continuous rendering loop
        this.startRenderLoop();
    }

    setupEventListeners() {
        // Range sliders with real-time updates
        document.getElementById('clusterTightness').addEventListener('input', (e) => {
            document.getElementById('clusterTightnessValue').textContent = parseFloat(e.target.value).toFixed(1);
            this.generateInitialData();
        });

        document.getElementById('ringWidth').addEventListener('input', (e) => {
            document.getElementById('ringWidthValue').textContent = parseFloat(e.target.value).toFixed(1);
            this.generateInitialData();
        });

        document.getElementById('mixingPercent').addEventListener('input', (e) => {
            document.getElementById('mixingPercentValue').textContent = e.target.value + '%';
            this.generateInitialData();
        });

        document.getElementById('numPoints').addEventListener('input', (e) => {
            document.getElementById('numPointsValue').textContent = e.target.value;
            this.generateInitialData();
        });

        document.getElementById('hiddenLayers').addEventListener('input', (e) => {
            document.getElementById('hiddenLayersValue').textContent = e.target.value;
            if (!this.isTraining) {
                this.resetNetwork();
            }
        });

        document.getElementById('neuronsPerLayer').addEventListener('input', (e) => {
            document.getElementById('neuronsPerLayerValue').textContent = e.target.value;
            if (!this.isTraining) {
                this.resetNetwork();
            }
        });

        document.getElementById('learningRate').addEventListener('input', (e) => {
            document.getElementById('learningRateValue').textContent = parseFloat(e.target.value).toFixed(3);
        });

        document.getElementById('epochs').addEventListener('input', (e) => {
            document.getElementById('epochsValue').textContent = e.target.value;
            this.totalEpochs = parseInt(e.target.value);
            document.getElementById('totalEpochs').textContent = this.totalEpochs;
        });

        document.getElementById('animationSpeed').addEventListener('input', (e) => {
            document.getElementById('animationSpeedValue').textContent = e.target.value;
        });

        // Buttons
        document.getElementById('trainButton').addEventListener('click', () => this.startTraining());
        document.getElementById('pauseButton').addEventListener('click', () => this.pauseTraining());
        document.getElementById('resetButton').addEventListener('click', () => this.resetNetwork());
    }

    startRenderLoop() {
        const render = () => {
            if (!this.isTraining) {
                this.draw();
            }
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    generateInitialData() {
        const numPoints = parseInt(document.getElementById('numPoints').value);
        const clusterTightness = parseFloat(document.getElementById('clusterTightness').value);
        const ringWidth = parseFloat(document.getElementById('ringWidth').value);
        const mixingPercent = parseInt(document.getElementById('mixingPercent').value);

        this.data = DataGenerator.generate(numPoints, clusterTightness, ringWidth, mixingPercent);
        if (!this.isTraining) {
            this.resetNetwork();
        }
    }

    resetNetwork() {
        const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
        const neuronsPerLayer = parseInt(document.getElementById('neuronsPerLayer').value);
        this.network = new NeuralNetwork(2, hiddenLayers, neuronsPerLayer, 1);
        this.currentEpoch = 0;
        this.totalEpochs = parseInt(document.getElementById('epochs').value);
        document.getElementById('currentEpoch').textContent = '0';
        document.getElementById('totalEpochs').textContent = this.totalEpochs;
        document.getElementById('currentLoss').textContent = '0.000';
        document.getElementById('currentAccuracy').textContent = '0.00%';
        this.drawNetworkArchitecture();
        this.draw();
    }

    async startTraining() {
        if (this.isTraining) return;
        if (!this.network) this.resetNetwork();

        this.isTraining = true;
        this.totalEpochs = parseInt(document.getElementById('epochs').value);
        document.getElementById('trainButton').disabled = true;
        document.getElementById('pauseButton').disabled = false;
        document.getElementById('trainingStatus').classList.add('training');
        document.getElementById('trainingStatus').classList.remove('complete');
        document.getElementById('trainingStatus').textContent = 'Training...';

        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const animationSpeed = parseInt(document.getElementById('animationSpeed').value);
        const delayMs = Math.max(1, 1000 / animationSpeed);

        const trainingLoop = async () => {
            for (let epoch = this.currentEpoch; epoch < this.totalEpochs && this.isTraining; epoch++) {
                // Shuffle data for better learning
                const shuffledData = [...this.data].sort(() => Math.random() - 0.5);
                
                let totalLoss = 0;
                let correctPredictions = 0;

                // Train on all data points
                for (let i = 0; i < shuffledData.length; i++) {
                    const point = shuffledData[i];
                    const input = [point.x, point.y];
                    this.network.backward(input, point.label, learningRate);
                    totalLoss += this.network.loss;

                    // Calculate accuracy
                    const prediction = this.network.predict(input);
                    const predicted = prediction > 0.5 ? 1 : 0;
                    if (predicted === point.label) {
                        correctPredictions++;
                    }
                }

                this.currentEpoch = epoch + 1;
                const avgLoss = totalLoss / shuffledData.length;
                const accuracy = (correctPredictions / shuffledData.length) * 100;

                document.getElementById('currentEpoch').textContent = this.currentEpoch;
                document.getElementById('currentLoss').textContent = avgLoss.toFixed(4);
                document.getElementById('currentAccuracy').textContent = accuracy.toFixed(2) + '%';

                this.draw();
                await new Promise(resolve => setTimeout(resolve, delayMs));
            }

            this.completeTraining();
        };

        trainingLoop();
    }

    completeTraining() {
        this.isTraining = false;
        document.getElementById('trainButton').disabled = false;
        document.getElementById('pauseButton').disabled = true;
        document.getElementById('trainingStatus').classList.remove('training');
        document.getElementById('trainingStatus').classList.add('complete');
        document.getElementById('trainingStatus').textContent = 'Training complete!';
    }

    pauseTraining() {
        this.isTraining = false;
        document.getElementById('trainButton').disabled = false;
        document.getElementById('pauseButton').disabled = true;
        document.getElementById('trainingStatus').classList.remove('training');
        document.getElementById('trainingStatus').textContent = 'Training paused.';
    }

    draw() {
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, width, height);

        // Draw decision boundary and gradient ONLY during or after training
        if (this.network && this.currentEpoch > 0) {
            const resolution = 4; // Balance between speed and smoothness
            const imageData = this.ctx.createImageData(width, height);
            const data = imageData.data;

            for (let i = 0; i < width; i += resolution) {
                for (let j = 0; j < height; j += resolution) {
                    const x = i / width;
                    const y = j / height;
                    const prediction = this.network.predict([x, y]);

                    // Color based on prediction (stronger colors for more confident predictions)
                    let r, g, b, a;

                    if (prediction > 0.5) {
                        // Red region (class 1)
                        const intensity = Math.pow(prediction, 0.7) * 255;
                        r = Math.floor(intensity);
                        g = Math.floor(50 * (1 - prediction));
                        b = Math.floor(50 * (1 - prediction));
                    } else {
                        // Blue region (class 0)
                        const intensity = Math.pow(1 - prediction, 0.7) * 255;
                        r = Math.floor(50 * prediction);
                        g = Math.floor(100 * (1 - prediction));
                        b = Math.floor(intensity);
                    }
                    a = 180;

                    // Fill rectangle
                    for (let di = 0; di < resolution && i + di < width; di++) {
                        for (let dj = 0; dj < resolution && j + dj < height; dj++) {
                            const idx = ((j + dj) * width + (i + di)) * 4;
                            data[idx] = r;
                            data[idx + 1] = g;
                            data[idx + 2] = b;
                            data[idx + 3] = a;
                        }
                    }
                }
            }

            this.ctx.putImageData(imageData, 0, 0);

            // Draw decision boundary line - simplified for performance
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 2.5;
            this.ctx.beginPath();

            let firstPoint = true;
            const step = 3;
            
            for (let x = 0; x <= width; x += step) {
                const xNorm = x / width;
                
                // Find boundary by sampling vertically
                let bestY = height / 2;
                let bestDiff = 1;
                
                for (let y = 0; y < height; y += 5) {
                    const yNorm = y / height;
                    const pred = this.network.predict([xNorm, yNorm]);
                    const diff = Math.abs(pred - 0.5);
                    
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        bestY = y;
                    }
                }

                if (firstPoint) {
                    this.ctx.moveTo(x, bestY);
                    firstPoint = false;
                } else {
                    this.ctx.lineTo(x, bestY);
                }
            }
            this.ctx.stroke();
        }

        // Draw data points (always visible)
        for (const point of this.data) {
            const x = point.x * width;
            const y = point.y * height;
            const radius = 5;

            this.ctx.fillStyle = point.color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.fill();

            // Draw border
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 1.5;
            this.ctx.stroke();
        }

        // Draw grid (always visible)
        this.ctx.strokeStyle = '#e8e8e8';
        this.ctx.lineWidth = 0.5;
        for (let i = 0; i <= width; i += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(i, 0);
            this.ctx.lineTo(i, height);
            this.ctx.stroke();
        }
        for (let j = 0; j <= height; j += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, j);
            this.ctx.lineTo(width, j);
            this.ctx.stroke();
        }
    }

    drawNetworkArchitecture() {
        if (!this.network) return;

        const width = this.networkCanvas.width;
        const height = this.networkCanvas.height;
        const ctx = this.networkCtx;

        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);

        const architecture = this.network.architecture;
        const layerCount = architecture.length;
        const layerWidth = width / (layerCount + 1);

        // Draw layers and neurons
        for (let i = 0; i < layerCount; i++) {
            const neuronCount = architecture[i];
            const layerX = layerWidth * (i + 1);
            const neuronHeight = (height - 20) / (Math.max(...architecture) + 1);

            // Draw neurons
            for (let j = 0; j < neuronCount; j++) {
                const neuronY = height / 2 - (neuronCount * neuronHeight) / 2 + j * neuronHeight;
                
                ctx.fillStyle = i === 0 ? '#667eea' : (i === layerCount - 1 ? '#ff6b6b' : '#764ba2');
                ctx.beginPath();
                ctx.arc(layerX, neuronY, 6, 0, 2 * Math.PI);
                ctx.fill();

                // Draw connections to next layer
                if (i < layerCount - 1) {
                    const nextNeuronCount = architecture[i + 1];
                    const nextLayerX = layerWidth * (i + 2);

                    for (let k = 0; k < nextNeuronCount; k++) {
                        const nextNeuronY = height / 2 - (nextNeuronCount * neuronHeight) / 2 + k * neuronHeight;
                        
                        ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(layerX, neuronY);
                        ctx.lineTo(nextLayerX, nextNeuronY);
                        ctx.stroke();
                    }
                }
            }

            // Draw layer label
            ctx.fillStyle = '#666';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            if (i === 0) {
                ctx.fillText('Input', layerX, 10);
            } else if (i === layerCount - 1) {
                ctx.fillText('Output', layerX, 10);
            } else {
                ctx.fillText('Hidden', layerX, 10);
            }
        }
    }
}

// ============================================================================
// Initialize Application
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
    new App();
});
