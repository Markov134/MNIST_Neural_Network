//
// Created by Steph on 11/16/2025.
//


#include "Net.h"

// ********************* class Neuron *********************

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connections());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
    m_outputVal = 0.0;
    m_gradient = 0.0;
}

// this part is broken
void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;

    // Sum the previous layer's output (which are our inputs)
    // Include the bias node from the previous layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        // we can get the value either through m_target or getTarget()
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    // transfer function is also static class member
    m_outputVal = Neuron::transferFunction(sum); // called either activation function or transfer function
}

double Neuron::transferFunction(double x) {
    //sigmoid function / [0.0, 1.0] outputs values 0-9
    return 1.0 / (1.0 + exp(-x));
}

// check to see if this equation is correct
double Neuron::transferFunctionDerivative(double x) {
    //sigmoid derivative
    return x * (1 - x);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
    //std::cout << m_gradient << std::endl;
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    // The weights to be updated are in the Connections container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        // Individual input, maginified by the gradient and train rate:
        double newDeltaWeight = eta // overall net learning rate
                                * neuron.getOutputVal()
                                * m_gradient
                                // Also add momentum = a fraction of the previous delta weight
                                + alpha // momentum
                                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::eta = 0.15; // overall net learning rate, [0.0..1.0]

double Neuron::alpha = 0.5; // momentum, multiplier of last delta weight, [0.0..n]

double Neuron::randomWeight(void) {
    return rand() / double(RAND_MAX) * 0.1 - 0.05;
}

// ********************* class Net *********************

Net::Net(const std::vector<unsigned> &topology) :m_error(0.0), m_recentAverageError(0.0) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We made a new layer, now fill it with neurons, and
        // add a bias neurons to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            // notice in the for loop we do <=. this is because we added a bias otherwise its just <.
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            //std::cout << "Made A Neuron!" << std::endl;
        }

        // Force the bias node's target value to 1.0. It is the last neuron created above
        m_layers.back().back().setOutputVal(1.0);
    }
    std::cout << "Finished Making Neurons!" << std::endl;
}

void Net::feedForward(const std::vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1); // checks for errors. Also the - 1 in size is for the bias neuron.
    // what it specifically does is it checks input is equal to the number of input neurons.

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward Propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals) {
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }

    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // implement a recent average measurement
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size() - 1; ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    // For all layers from outputs to first hidden layer,
    // update the connection weights.
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultsVals) const {
    resultsVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() -1 ; ++n) {
        resultsVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

double Net::m_recentAverageSmoothingFactor = 100.0;

// ********************* class TrainingData *********************

TrainingData::TrainingData(std::vector<Data*>* dataArray) : m_data(dataArray), m_index(0), numClasses(0) {}

bool TrainingData::isEof() const {
    return m_index >= m_data->size();
}

void TrainingData::reset() {
    m_index = 0;
}

unsigned TrainingData::getNextInputs(std::vector<double> &input) {
    if (isEof())
        return 0;

    input = (*m_data)[m_index]->toInputVector();
    m_index++;
    return input.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &target) {
    if (isEof())
        return 0;

    target = (*m_data)[m_index]->toTargetVector(numClasses);
    m_index++;
    return target.size();
}

void TrainingData::setNumClasses(int n) { numClasses = n; }

void showVectorVals(std::string label, std::vector<double> &v) {
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}