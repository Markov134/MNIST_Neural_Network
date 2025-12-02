#include <iostream>
#include "src/Data.h"
#include "src/DataLoader.h"
#include "src/Net.h"
#include <algorithm>


//use MNIST dataset for the images
int main() {

    // Load the data
    // *************************************
    DataLoader loader;
    loader.read_feature_vector("Files/train-images.idx3-ubyte");
    loader.read_feature_labels("Files/train-labels.idx1-ubyte");
    loader.countClasses();
    loader.splitData();

    auto trainingData = loader.getTrainingData();
    auto testData = loader.getTestData();

    unsigned inputSize = trainingData->at(0)->get_feature_vector_size();
    unsigned outputSize = loader.getNumClasses();
    int numClasses = outputSize;

    // Build neural network
    // *************************************
    std::vector<unsigned> topology;
    topology.push_back(inputSize);
    topology.push_back(64); // hidden layer
    topology.push_back(outputSize);

    Net myNet(topology);

    // Training
    // *************************************
    int epochs = 1;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::cout << "Starting epoch " << epoch << std::endl;

        for (size_t i = 0; i < trainingData->size(); i++) {
            Data* sample = trainingData->at(i);

            std::vector<double> inputVals  = sample->toInputVector();
            std::vector<double> targetVals = sample->toTargetVector(numClasses);

            myNet.feedForward(inputVals);
            myNet.backProp(targetVals);
        }

        std::cout << "Finished epoch " << epoch
                  << " | Avg Error = " << myNet.getRecentAverageError()
                  << "\n";
    }

    // Testing
    // *************************************
    int N = 11;   // number of tests
    int correct = 0;
    std::cout << "\n=== TESTING FIRST " << N << " TEST IMAGES ===\n";

    for (int i = 0; i < N; i++) {
        Data* sample = testData->at(i);

        std::vector<double> input = sample->toInputVector();
        myNet.feedForward(input);

        std::vector<double> output;
        myNet.getResults(output);

        auto it = std::max_element(output.begin(), output.end());
        int predicted = it - output.begin();

        int actual = sample->get_label();

        if (predicted == actual)
            correct++;

        std::cout << "Sample " << i + 1
                  << " | Actual: " << actual
                  << " | Predicted: " << predicted << "\n";
    }

    double accuracy = 100.0 * correct / N;
    std::cout << " Accuracy: " << accuracy << "%\n";
}