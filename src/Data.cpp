//
// Created by Steph on 11/1/2025.
//

#include "Data.h"

Data::Data() {
    feature_vector = new std::vector<uint8_t>;
}

Data::~Data() {
    delete feature_vector;
}

void Data::set_feature_vector(std::vector<uint8_t> *vect) {
    feature_vector = vect;
}

void Data::append_to_feature_vector(uint8_t val) {
    feature_vector->push_back(val);
}

void Data::set_label(uint8_t val) {
    label = val;
}
void Data::set_enum_label(int val) {
    enum_label = val;
}

int Data::get_feature_vector_size() {
    return feature_vector->size();
}

uint8_t Data::get_label() {
    return label;
}

uint8_t Data::get_enum_label() {
    return enum_label;
}

std::vector<uint8_t> * Data::get_feature_vector() {
    return feature_vector;
}

std::vector<double> Data::toInputVector() const {
    std::vector<double> input;
    input.reserve(feature_vector->size());

    for (uint8_t pixel : *feature_vector) {
        input.push_back(pixel / 255.0);
    }

    return input;
}

std::vector<double> Data::toTargetVector(int numClasses) const {
    std::vector<double> target(numClasses, 0.0);
    if (enum_label >= 0 && enum_label < numClasses) {
        target[enum_label] = 1.0;
    }

    return target;
}
