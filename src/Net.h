//
// Created by Steph on 11/16/2025.
//

#ifndef NET_H
#define NET_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include "Data.h"

class TrainingData {
    public:
    TrainingData(std::vector<Data*>* dataArray);
    bool isEof() const;
    void reset();

    // Returns the number of input values read from the file:
    unsigned getNextInputs(std::vector<double> &input);
    unsigned getTargetOutputs(std::vector<double> &target);
    void setNumClasses(int n);
private:
    std::vector<Data*>* m_data;
    unsigned m_index;
    int numClasses;
};

struct Connections {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta; // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connections> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};


class Net {
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultsVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;

};

void showVectorVals(std::string label, std::vector<double> &v);
#endif //NET_H
