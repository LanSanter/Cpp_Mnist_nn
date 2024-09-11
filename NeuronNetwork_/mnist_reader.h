#pragma once
// mnist_reader.h

#include <vector>
#include <string>

struct IDXHeader {
    uint32_t magic_number;
    uint32_t num_images;
    uint32_t rows;
    uint32_t cols;
    size_t image_size;
};

// declaim function
bool readMNISTImages(const char* filename, std::vector<std::vector<uint8_t>>& images);
bool readMNISTLabels(const char* filename, std::vector<uint8_t>& labels);