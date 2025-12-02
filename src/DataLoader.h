//
// Created by Steph on 11/1/2025.
//

#ifndef DATALOADER_H
#define DATALOADER_H
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include "data.h"
#include <cstdlib>
#include <algorithm>
#include <random>

class DataLoader {
private:
    std::vector<Data *> * dataArr;
    std::vector<Data *> * trainingData;
    std::vector<Data *> * testData;
    std::vector<Data *> * validationData;

    int numClasses;
    int feature_vector_size;
    std::map<uint8_t, int> classMap;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_PERCENT = 0.05;

public:
    DataLoader();
    ~DataLoader();

    void read_feature_vector(std::string filename);
    void read_feature_labels(std::string filename);
    void splitData();
    void countClasses();
    int getNumClasses() const;

    uint32_t converToLittleEndian(const unsigned char* bytes);

    std::vector<Data *> * getTrainingData();
    std::vector<Data *> * getTestData();
    std::vector<Data *> * getValidationData();
};

#endif //DATALOADER_H
