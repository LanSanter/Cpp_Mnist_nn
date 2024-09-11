#include <iostream>
#include <fstream>
#include <vector>
#include "mnist_reader.h"
#include <winsock2.h>
#include <algorithm>
#include "zlib.h"


IDXHeader readIDX3Header(std::ifstream& file) {
    IDXHeader header;
    file.read(reinterpret_cast<char*>(&header.magic_number), sizeof(header.magic_number));
    header.magic_number = ntohl(header.magic_number);
    file.read(reinterpret_cast<char*>(&header.num_images), sizeof(header.num_images));
    header.num_images = ntohl(header.num_images);
    file.read(reinterpret_cast<char*>(&header.rows), sizeof(header.rows));
    header.rows = ntohl(header.rows);
    file.read(reinterpret_cast<char*>(&header.cols), sizeof(header.cols));
    header.cols = ntohl(header.cols);
    header.image_size = header.rows * header.cols;
    return header;
}

IDXHeader readIDX1Header(std::ifstream& file) {
    IDXHeader header;
    file.read(reinterpret_cast<char*>(&header.magic_number), sizeof(header.magic_number));
    header.magic_number = ntohl(header.magic_number);
    file.read(reinterpret_cast<char*>(&header.num_images), sizeof(header.num_images));
    header.num_images = ntohl(header.num_images);
    header.rows = 0;
    header.cols = 0;
    return header;
}

bool readMNISTLabels(const char* filename, std::vector<uint8_t>& labels) {

    gzFile gzfile = gzopen(filename, "rb");
    if (!gzfile) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        return false;
    }

    std::vector<char> buffer(1024);

    std::ofstream temp_file("temp.idx", std::ios::binary);
    if (!temp_file.is_open()) {
        std::cerr << "Unable to open output file." << std::endl;
        gzclose(gzfile);
        return false;
    }

    int n_bytes_read;
    while ((n_bytes_read = gzread(gzfile, buffer.data(), buffer.size())) > 0) {
        temp_file.write(buffer.data(), n_bytes_read);
    }
    gzclose(gzfile);
    temp_file.close();


    std::ifstream idx_file("temp.idx", std::ios::binary);
    if (!idx_file.is_open()) {
        std::cerr << "Error: Failed to open IDX file" << std::endl;
        return false;
    }


    IDXHeader header_ = readIDX1Header(idx_file);
    idx_file.read(reinterpret_cast<char*>(labels.data()), header_.num_images);

    if (idx_file.gcount() != header_.num_images) {
        std::cerr << "Error: Failed to read label data for image " << idx_file.gcount() << std::endl;
        return false;
    }
    std::cout << "Magic Number: " << header_.magic_number << std::endl;
    std::cout << "Number of Images: " << header_.num_images << std::endl;
    /*for (size_t i = 0; i < 10; ++i) {
        std::cout << "label:" << labels[static_cast<int>(i)] << std::endl;
    }*/
    

    idx_file.clear();
    idx_file.close();
    remove("temp.idx");

    return true;
}

bool readMNISTImages(const char* filename, std::vector<std::vector<uint8_t>>& images) {
    //const char* input_filename = "D:\\NeuronNetwork\\mnist\\train-images-idx3-ubyte.gz";

    gzFile gzfile= gzopen(filename, "rb");
    if (!gzfile) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        return false;
    }

    
    /*
        image_data
                       */

    std::vector<char> buffer(1024);

    std::ofstream temp_file("temp.idx", std::ios::binary);
    if (!temp_file.is_open()) {
        std::cerr << "Unable to open output file." << std::endl;
        gzclose(gzfile);
        return false;
    }

    int num_bytes_read;
    while ((num_bytes_read = gzread(gzfile, buffer.data(), buffer.size())) > 0) {
        temp_file.write(buffer.data(), num_bytes_read);
    }
    gzclose(gzfile);
    temp_file.close();


    std::ifstream idx_file("temp.idx", std::ios::binary);
    if (!idx_file.is_open()) {
        std::cerr << "Error: Failed to open IDX file" << std::endl;
        return false;
    }


    IDXHeader header = readIDX3Header(idx_file);
    for (uint32_t i = 0; i < header.num_images; ++i) {
        std::vector<uint8_t> single_image(header.image_size);
        idx_file.read(reinterpret_cast<char*>(single_image.data()), header.image_size);

        if (idx_file.gcount() != header.image_size) {
            std::cerr << "Error: Failed to read image data for image " << i << std::endl;
            return false;
        }
        images.push_back(single_image);
    }

    idx_file.clear();
    idx_file.close();
    remove("temp.idx");

    //std::cout << "Magic Number: " << header.magic_number << std::endl;
    //std::cout << "Number of Images: " << header.num_images << std::endl;
    //std::cout << "Image Size: " << header.rows << "x" << header.cols << std::endl;





    return true;
}