import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {

    public enum WEIGHT_STRATEGY {
        RANDOM, FIXED;
    }
    private WEIGHT_STRATEGY weightStrategy;
    private static final Random r = new Random();

    private int inputFeatureCount;
    private int hiddenLayerSize;
    private int outputFeatureCount;

    // Weights of edges which are inbound to the hidden layer
    // Example: [2][4] represents the edge from the 3rd feature to the 5th hidden node
    private double[][] hiddenLayerWeights;
    private double[] hiddenLayerBias;

    // Tracks the last change we made to the weights - combined with momentum when training
    private double[][] hiddenLayerWeightLastChange;
    private double[] hiddenLayerBiasLastChange;

    private double[][] outputLayerWeights;
    private double[] outputLayerBias;

    private double[][] outputLayerWeightLastChange;
    private double[] outputLayerBiasLastChange;

    private double learningRate;
    private double momentum;

    public NeuralNetwork(int inputFeatureCount, int hiddenNodes, int outputFeatureCount,
                         double learningRate, double momentum, WEIGHT_STRATEGY strategy) {

        this.weightStrategy = strategy;
        this.inputFeatureCount = inputFeatureCount;


        this.hiddenLayerSize = hiddenNodes;
        this.hiddenLayerWeights = initializeWeightArray(inputFeatureCount, hiddenNodes, false);
        this.hiddenLayerBias = initializeBias(hiddenNodes, false);

        this.hiddenLayerWeightLastChange = initializeWeightArray(inputFeatureCount, hiddenNodes, true);
        this.hiddenLayerBiasLastChange = initializeBias(hiddenNodes, true);

        this.outputFeatureCount = outputFeatureCount;
        this.outputLayerWeights = initializeWeightArray(hiddenNodes, outputFeatureCount, false);
        this.outputLayerBias = initializeBias(outputFeatureCount, false);

        this.outputLayerWeightLastChange = initializeWeightArray(hiddenNodes, outputFeatureCount, true);
        this.outputLayerBiasLastChange = initializeBias(outputFeatureCount, true);

        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    // Run (and train) the model once
    public double[] Run(TrainingData trainingData) {
        return this.Run(trainingData, true);
    }

    public double[] Run(TrainingData trainingData, boolean shouldTrain) {
        double[] input = trainingData.inputs;
        double[] expectedOutput = trainingData.outputs;

        double[] hiddenOutput = getLayerOutput(input, hiddenLayerWeights, hiddenLayerBias);
        double[] actualOutput = getLayerOutput(hiddenOutput, outputLayerWeights, outputLayerBias);

        if (shouldTrain) {
            // Calculate d_pk
            double[] outputLayerDeltas = new double[outputFeatureCount];
            for (int i = 0; i < outputFeatureCount; i++) {
                double target = expectedOutput[i];
                double actual = actualOutput[i];
                double delta = (target - actual) * actual * (1 - actual);
                outputLayerDeltas[i] = delta;
            }

            // Calculate d_pj
            double[] hiddenLayerDeltas = new double[hiddenLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++) {
                double outputLayerDelta = 0;
                for (int j = 0; j < outputFeatureCount; j++) {
                    outputLayerDelta += outputLayerDeltas[j] * outputLayerWeights[i][j];
                }
                double actual = hiddenOutput[i];
                double delta = actual * (1 - actual) * outputLayerDelta;
                hiddenLayerDeltas[i] = delta;
            }

            updateLayer(hiddenLayerDeltas, input, hiddenLayerWeights, hiddenLayerBias, hiddenLayerWeightLastChange, hiddenLayerBiasLastChange);
            updateLayer(outputLayerDeltas, hiddenOutput, outputLayerWeights, outputLayerBias, outputLayerWeightLastChange, outputLayerBiasLastChange);
        }

        return actualOutput;
    }

    public double GetError(ArrayList<TrainingData> trainingData) {
        double error = 0.0;

        for (TrainingData data : trainingData) {
            double[] actualOutput = this.Run(data, false);
            error += getTotalError(data.outputs, actualOutput);
        }

        return error;
    }

    // Code to update a layer's weights, bias, etc.
    private void updateLayer(double[] deltas, double[] previousLayerOutput, double[][] weightsToUpdate, double[] biasToUpdate,
                                 double[][] lastWeightChange, double[] lastBiasChange) {
        for (int node = 0; node < deltas.length; node++) {
            double changeInBias = learningRate * deltas[node];
            changeInBias += momentum * lastBiasChange[node];
            biasToUpdate[node] += changeInBias;
            lastBiasChange[node] = changeInBias;

            for (int previousNode = 0; previousNode < previousLayerOutput.length; previousNode++) {
                double changeInWeight = learningRate * deltas[node] * previousLayerOutput[previousNode];
                changeInWeight += momentum * lastWeightChange[previousNode][node];
                weightsToUpdate[previousNode][node] += changeInWeight;
                lastWeightChange[previousNode][node] = changeInWeight;
            }
        }
    }

    // Creates a 2D array with the specified dimensions and starting values.
    private double[][] initializeWeightArray(int numSources, int numSinks, boolean zero) {
        double[][] weights = new double[numSources][numSinks];
        for (int i = 0; i < numSources; i++) {
            for (int j = 0; j < numSinks; j++) {
                weights[i][j] = this.getStartingWeight(zero);
            }
         }
        return weights;
    }

    // Creates an array with the specified dimension and starting value
    private double[] initializeBias(int numNodes, boolean zero) {
        double[] bias = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            bias[i] = this.getStartingWeight(zero);
        }
        return bias;
    }

    // Calculate the output of a layer.
    private double[] getLayerOutput(double[] layerInput, double[][] layerWeights, double[] layerBias) {
        int sizeOfPreviousLayer = layerInput.length;
        int sizeOfLayer = layerWeights[0].length;
        double[] output = new double[sizeOfLayer];

        for (int i = 0; i < sizeOfLayer; i++) {
            double net = 0;
            for (int j = 0; j < sizeOfPreviousLayer; j++) {
                net += layerInput[j] * layerWeights[j][i];
            }
            net += layerBias[i];
            output[i] = sigmoid(net);
        }

        return output;
    }

    // Calculate total squared error of an output vector, vs the expected output
    private double getTotalError(double[] expectedOutput, double[] actualOutput) {
        double total = 0;
        for (int i = 0; i < expectedOutput.length; i++) {
            total += (Math.pow(expectedOutput[i] - actualOutput[i], 2) / 2);
        }
        return total;
    }

    // Sigmoid function
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-1 * x));
    }

    private double getStartingWeight(boolean zero) {
        if (zero)
            return 0;
        if (this.weightStrategy == WEIGHT_STRATEGY.FIXED)
            return 0.5;
        return r.nextDouble();
    }

    public static class TrainingData {
        public double[] inputs;
        public double[] outputs;
        public TrainingData(int inputSize, int outputSize) {
            this.inputs = new double[inputSize];
            this.outputs = new double[outputSize];
        }
    }
}
