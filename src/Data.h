//
// Created by Steph on 11/1/2025.
//

#ifndef DATA_H
#define DATA_H
#include <cstdint>
#include <vector>

class Data {
private:
    std::vector<uint8_t> * feature_vector;
    uint8_t label;
    uint8_t enum_label;

public:
    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t> * vect);
    void append_to_feature_vector(uint8_t val);
    void set_label(uint8_t val);
    void set_enum_label(uint8_t val);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enum_label();

    std::vector<uint8_t> * get_feature_vector();
    std::vector<double> toInputVector() const;
    std::vector<double> toTargetVector(int numClasses) const;
};

#endif //DATA_H
