//
// Created by Steph on 12/5/2025.
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "../src/Data.h"
#include "../src/DataLoader.h"
#include "../src/Net.h"

TEST_CASE(" BenchMarking The Nerual Network. ") {
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

    BENCHMARK(" FeedForward Trial 1 ") {
            Data* sample = trainingData->at(rand() % trainingData->size());

            std::vector<double> inputVals  = sample->toInputVector();
            return myNet.feedForward(inputVals);
    };

    BENCHMARK(" FeedForward Trial 2 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        return myNet.feedForward(inputVals);
    };

    BENCHMARK(" FeedForward Trial 3 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        return myNet.feedForward(inputVals);
    };

    BENCHMARK(" FeedForward Trial 4 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        return myNet.feedForward(inputVals);
    };

    BENCHMARK(" FeedForward Trial 5 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        return myNet.feedForward(inputVals);
    };

    BENCHMARK(" Backprop Trial 1 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" Backprop Trial 2 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" Backprop Trial 3 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" Backprop Trial 4 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" Backprop Trial 5 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" FeedForward + Backprop Trial 1 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        myNet.feedForward(inputVals);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" FeedForward + Backprop Trial 2 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        myNet.feedForward(inputVals);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" FeedForward + Backprop Trial 3 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        myNet.feedForward(inputVals);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" FeedForward + Backprop Trial 4 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        myNet.feedForward(inputVals);
        return myNet.backProp(targetVals);
    };

    BENCHMARK(" FeedForward + Backprop Trial 5 ") {
        Data* sample = trainingData->at(rand() % trainingData->size());

        std::vector<double> inputVals  = sample->toInputVector();
        std::vector<double> targetVals = sample->toTargetVector(numClasses);
        myNet.feedForward(inputVals);
        return myNet.backProp(targetVals);
    };
}

