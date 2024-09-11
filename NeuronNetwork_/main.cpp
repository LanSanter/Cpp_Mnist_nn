#include "mnist_reader.h"
#include "Neuron_layer.h"
#include <cmath>
#include <iostream>


void preprocess_batches(const std::vector<Eigen::VectorXd>& norm_vecs,
    const std::vector<uint8_t>& labels,
    size_t batch_size,
    std::vector<Eigen::MatrixXd>& batch_images,
    std::vector<Eigen::VectorXi>& batch_labels);

Eigen::VectorXi maxIndices(const Eigen::MatrixXd& output, const int& batch_size);
double calculate_accuracy(const Eigen::VectorXi& predict, const Eigen::VectorXi& batch_labels);

int main() {
    double learning_rate = 0.3;
    int epoch = 2;
    int batch_size = 100;
    std::vector<Eigen::MatrixXd> batch_images;
    std::vector<Eigen::VectorXi> batch_labels;
    std::string weight_file = "wieghts.bin";
    const char* image_file = "..\\mnist\\train-images-idx3-ubyte.gz";
    const char* label_file = "..\\mnist\\train-labels-idx1-ubyte.gz";
    std::vector <std::vector<uint8_t>>images;
    std::vector <uint8_t> labels(60000);
    bool  success1 = readMNISTImages(image_file, images);
    bool  success2 = readMNISTLabels(label_file, labels);
    if (!success1) {
        std::cout << "Unable to open image.gz file" << std::endl;
        return 1;
    }
    else if (!success2) {
        std::cout << "Unable to open label.gz file" << std::endl;
        return 1;
    }

    srand((unsigned int)time(0));

    

    std::vector<int> layer_size = {784, 1000, 10};
    std::vector<std::string> activations = {"relu", "softmax"};

    NeuralNetwork net(layer_size, activations); //create a neuron network
    net.load_weight(weight_file);

    size_t quantity = images.size();
    size_t size = images[0].size();
    std::vector<Eigen::VectorXd> norm_vecs(60000, Eigen::VectorXd(784));

    for (size_t i = 0; i < quantity; ++i) {
        for (size_t j = 0; j < size; ++j) {
            norm_vecs[static_cast<int>(i)][static_cast<int>(j)] = static_cast<double>(images[i][j]) / 255; // image_data in 0~255
        }
    }
   
    preprocess_batches(norm_vecs, labels, batch_size, batch_images, batch_labels);
    int round = quantity / batch_size;


    for (int j = 0; j < epoch; ++j) {
        for (int i = 0; i < round ; ++i) {
 
            Eigen::MatrixXd output = net.predict(batch_images[i]);
            net.backward(batch_labels[i]);
            net.update_weights_and_bias(learning_rate, batch_size);
            
            if ((i + 1) % 50 == 0) {
                Eigen::VectorXi predict = maxIndices(output, batch_size);
                double accuracy = calculate_accuracy(predict, batch_labels[i]);
                std::cout << "---------------" << std::endl;
                std::cout << "Epoch:" << (j + 1) << std::endl;
                std::cout << "Round:" << (i + 1) << std::endl;
                std::cout << "Accuracy: \n" << accuracy << std::endl;
                //std::cout << "Network output: \n" <<  predict << std::endl;
                //std::cout << "Labels:" << batch_labels[i] <<  std::endl;
                std::cout << "---------------" << std::endl;
                //std::cout << "Weight: \n" << net.layers[1].weights << std::endl;
            }
            
        }

    }

    net.save_weight(weight_file);

    return 0;
}

void preprocess_batches(const std::vector<Eigen::VectorXd>& norm_vecs,
    const std::vector<uint8_t>& labels,
    size_t batch_size,
    std::vector<Eigen::MatrixXd>& batch_images,
    std::vector<Eigen::VectorXi>& batch_labels) {
    size_t total_samples = norm_vecs.size();
    size_t num_batches = (total_samples + batch_size - 1) / batch_size; // Compute the number of batches

    batch_images.resize(num_batches);
    batch_labels.resize(num_batches);

    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start_index = batch * batch_size;
        size_t end_index = std::min(start_index + batch_size, total_samples);

        // Initialize batch_images and batch_labels
        batch_images[batch] = Eigen::MatrixXd(norm_vecs[0].size(), batch_size);
        batch_labels[batch] = Eigen::VectorXi(batch_size);

        for (size_t i = start_index; i < end_index; ++i) {
            size_t batch_index = i - start_index;
            batch_images[batch].col(batch_index) = norm_vecs[i]; // Assign image data
            batch_labels[batch](batch_index) = labels[i]; // Assign label
        }

        // If the batch is not full, we need to resize to the actual number of samples in this batch
        if (end_index - start_index < batch_size) {
            batch_images[batch].conservativeResize(end_index - start_index, Eigen::NoChange);
            batch_labels[batch].conservativeResize(end_index - start_index);
        }
    }
}

Eigen::VectorXi maxIndices(const Eigen::MatrixXd& output, const int& batch_size) {
    Eigen::VectorXi maxIndices(batch_size);

    for (int col = 0; col < batch_size; ++col) {

        double maxVal = output.col(col).maxCoeff();

        for (int row = 0; row < output.rows(); ++row) {
            if (output(row, col) == maxVal) {
                maxIndices(col) = row;
                break;
            }
        }
    }
    return maxIndices;
}

double calculate_accuracy(const Eigen::VectorXi& predict, const Eigen::VectorXi& batch_labels) {

    assert(predict.size() == batch_labels.size());


    int hit_num = 0;


    for (int i = 0; i < predict.size(); ++i) {
        if (predict[i] == batch_labels[i]) {
            hit_num += 1;
        }
    }


    double accuracy = static_cast<double>(hit_num) / predict.size();

    return accuracy;
}