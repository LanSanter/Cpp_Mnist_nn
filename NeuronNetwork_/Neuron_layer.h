#ifndef NEURON_LAYER_H  
#define NEURON_LAYER_H 

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <random>

Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits);
Eigen::MatrixXd softmax_crossentropy_gradient(const Eigen::MatrixXd& predict, const Eigen::VectorXi& one_hot);

Eigen::MatrixXd normalRandomMatrix(int rows, int cols, double mean, double stddev);
Eigen::VectorXd normalRandomVector(int size, double mean, double stddev);

class Layer {
public:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::string activation;
    Eigen::MatrixXd finput;  // Store last_layer_output*weight
    Eigen::MatrixXd weights_gradients;// store gradients for weight
    Eigen::VectorXd bias_gradients; // Store gradients for biases

    Layer(int input_size, int output_size, std::string activation_func);
    Eigen::MatrixXd apply_activation(const Eigen::MatrixXd& x);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    void update_gradients(const double& learning_rate, const int& batch_size);
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    std::vector<Eigen::MatrixXd> output_data;

    NeuralNetwork(const std::vector<int>& layer_sizes, const std::vector<std::string>& activations);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& input);
    void backward(const Eigen::VectorXi& target);
    void update_weights_and_bias(const double& learning_rate, const int& batch_size);
    void save_weight(const std::string& filename);
    void load_weight(const std::string& filename);
};



#endif
