//
// Created by Steph on 11/1/2025.
//

#include "DataLoader.h"

DataLoader::DataLoader() {
    dataArr = new std::vector<Data *>;
    trainingData = new std::vector<Data *>;
    testData = new std::vector<Data *>;
    validationData = new std::vector<Data *>;
}

DataLoader::~DataLoader() {
    for (auto d : *dataArr)
        delete d;
    delete dataArr;
    delete trainingData;
    delete testData;
    delete validationData;
}

void DataLoader::read_feature_vector(std::string filename) {
    uint32_t header[4]; //|MAGIC|NUM SIZE|ROWSIZE|COLSIZE|
    unsigned char bytes[4];
    FILE *f = fopen(filename.c_str(), "rb");

    if (f) {
        for (int i = 0; i < 4; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = converToLittleEndian(bytes);
            }
        }
        printf("Done getting input file header.\n");

        int image_size = header[2] * header[3];

        for (int i = 0; i < header[1]; i++) {

            Data *d = new Data();
            uint8_t element[1];
            //image size iteration
            for (int j = 0; j < image_size; j++) {
                if (fread(element, sizeof(element), 1, f)) {
                    d->append_to_feature_vector(element[0]);
                } else {
                    printf("Error while reading file header.\n");
                    fclose(f);
                    exit(1);
                }
            }
            dataArr->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", dataArr->size());
        fclose(f);
    } else {
        printf("Error opening file.\n");
        fclose(f);
        exit(1);
    }
}

void DataLoader::read_feature_labels(std::string filename) {

    uint32_t header[2]; //|MAGIC|NUM SIZE
    unsigned char bytes[4];
    FILE *f = fopen(filename.c_str(), "rb");

    if (f) {
        for (int i = 0; i < 2; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = converToLittleEndian(bytes);
            }
        }
        printf("Done getting label file header.\n");

        for (int i = 0; i < header[1]; i++) {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f)) {
                dataArr->at(i)->set_label(element[0]);
            } else {
                printf("Error while reading file header.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored label.\n");
    } else {
        printf("Error opening file.\n");
        exit(1);
    }
}

void DataLoader::splitData() {
    std::vector<int> indices(dataArr->size());
    std::iota(indices.begin(), indices.end(), 0); // fill [0, 1, 2, ... , N-1]

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    int trainSize = dataArr->size() * TRAIN_SET_PERCENT;
    int testSize = dataArr->size() * TEST_SET_PERCENT;
    int validationSize = dataArr->size() * VALIDATION_PERCENT;

    // Training data
    for (int i = 0; i < trainSize; i++) {
        trainingData->push_back(dataArr->at(indices[i]));
    }

    // Test data
    for (int i = trainSize; i < trainSize + testSize; i++) {
        testData->push_back(dataArr->at(indices[i]));
    }

    // Validation data
    for (int i = trainSize + testSize; i < trainSize + testSize + validationSize; i++) {
        validationData->push_back(dataArr->at(indices[i]));
    }

    printf("Training Data Size: %lu\n", trainingData->size());
    printf("Test Data Size: %lu\n", testData->size());
    printf("Validation Data Size: %lu\n", validationData->size());
}

void DataLoader::countClasses() {
    numClasses = 10;

    for (unsigned i = 0; i < dataArr->size(); i++) {
        uint8_t label = dataArr->at(i)->get_label();
        dataArr->at(i)->set_enum_label(label); // enum_label matches MNIST label
    }

    printf("Successfully set enum_labels for %d classes.\n", numClasses);
}

uint32_t DataLoader::converToLittleEndian(const unsigned char* bytes) {
    return (uint32_t) (bytes[0] << 24 |
                       bytes[1] << 16 |
                       bytes[2] << 8 |
                       bytes[3]);
}

std::vector<Data *> * DataLoader::getTrainingData() {
    return trainingData;
}

std::vector<Data *> * DataLoader::getTestData() {
    return testData;
}

std::vector<Data *> * DataLoader::getValidationData() {
    return validationData;
}

int DataLoader::getNumClasses() const {
    return numClasses;
}