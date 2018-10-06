import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.*;
import org.jfree.chart.*;
import org.jfree.util.ShapeUtilities;

import java.awt.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;


public class Main {
    public static void main(String[] args) {
        (new Main()).Go();
    }

    public void Go() {
        // Configure parameters:
        int numInputFeatures = 2;
        int numOutputFeatures = 3;
        int numHiddenNodes = 20;

        double learningRate = 0.5;
        double momentum = 0.8;

        int numIterations = 30000;

        NeuralNetwork n = new NeuralNetwork(numInputFeatures, numHiddenNodes, numOutputFeatures,
                learningRate, momentum, NeuralNetwork.WEIGHT_STRATEGY.RANDOM);
        ArrayList<NeuralNetwork.TrainingData> data = this.readStdIn(numInputFeatures, numOutputFeatures);

        System.out.println("Initial error: " + n.GetError(data));

        for (int iteration = 1; iteration <= numIterations; iteration++) {
            for (NeuralNetwork.TrainingData dataPoint : data) {
                n.Run(dataPoint);
            }

            if (iteration == 1 || iteration == 3000 || iteration == numIterations) {
                double score = n.GetError(data);
                System.out.printf("After %d iterations, error is %f\n", iteration, score);
            }
        }


        // For now, this is hard-coded for a 2x3 system.
        XYSeriesCollection results = new XYSeriesCollection();
        XYSeries[] series = new XYSeries[6];
        series[0] = new XYSeries("Type 1 (Correct)");
        series[1] = new XYSeries("Type 1 (Incorrect)");
        series[2] = new XYSeries("Type 2 (Correct)");
        series[3] = new XYSeries("Type 2 (Incorrect)");
        series[4] = new XYSeries("Type 3 (Correct)");
        series[5] = new XYSeries("Type 3 (Incorrect)");

        for (NeuralNetwork.TrainingData dataPoint : data) {
            int classification = getClassification(n.Run(dataPoint, false));
            int bucket = classification * 2;
            if (classification != getClassification(dataPoint.outputs))
                bucket++;
            series[bucket].add(dataPoint.inputs[0], dataPoint.inputs[1]);
        }

        results.addSeries(series[0]);
        results.addSeries(series[1]);
        results.addSeries(series[2]);
        results.addSeries(series[3]);
        results.addSeries(series[4]);
        results.addSeries(series[5]);

        // create a chart...
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Experiment Results", "X", "Y", results, PlotOrientation.VERTICAL, true, true, false);

        XYItemRenderer renderer = ((XYPlot)chart.getPlot()).getRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesPaint(1, Color.RED);
        renderer.setSeriesShape(1, ShapeUtilities.createDiagonalCross(3, 1));
        renderer.setSeriesPaint(2, Color.GREEN);
        renderer.setSeriesPaint(3, Color.GREEN);
        renderer.setSeriesShape(3, ShapeUtilities.createDiagonalCross(3, 1));
        renderer.setSeriesPaint(4, Color.BLUE);
        renderer.setSeriesPaint(5, Color.BLUE);
        renderer.setSeriesShape(5, ShapeUtilities.createDiagonalCross(3, 1));

        // create and display a frame...
        ChartFrame frame = new ChartFrame("Frame", chart);
        frame.pack();
        frame.setVisible(true);
    }

    private ArrayList<NeuralNetwork.TrainingData> readStdIn(int inputSize, int outputSize) {
        Scanner in = new Scanner(System.in);
        ArrayList<NeuralNetwork.TrainingData> dataset = new ArrayList<>();

        while (in.hasNextDouble()) {
            NeuralNetwork.TrainingData data = new NeuralNetwork.TrainingData(inputSize, outputSize);
            for (int i = 0; i < inputSize; i++) {
                data.inputs[i] = in.nextDouble();
            }
            for (int i = 0; i < outputSize; i++) {
                data.outputs[i] = in.nextDouble();
            }
            dataset.add(data);
        }

        return dataset;
    }

    private int getClassification(double[] results) {
        int classification = 0;

        if (results[1] > results[0] && results[1] > results[2])
            classification = 1;
        if (results[2] > results[0] && results[2] > results[1])
            classification = 2;

        return classification;
    }
}
