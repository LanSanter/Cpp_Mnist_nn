#include "Neuron_layer.h"




// Softmax ¨ç¼Æ
Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits) {

    Eigen::MatrixXd scaled_logits = logits.rowwise() - logits.colwise().maxCoeff();// length 64 vector

    Eigen::MatrixXd exp_values = scaled_logits.array().exp();

    Eigen::VectorXd col_sums = exp_values.colwise().sum();

    Eigen::MatrixXd softmax_values = exp_values.array().rowwise() / col_sums.array().transpose();

    return softmax_values;
}

// Softmax Cross-Entropy ±è«×­pºâ
Eigen::MatrixXd softmax_crossentropy_gradient(const Eigen::MatrixXd& predict, const Eigen::VectorXi& one_hot) {
    Eigen::MatrixXd gradient = predict;
    for (int i = 0; i < gradient.rows(); ++i) {
        gradient(i, one_hot[i]) -= 1;
    }
    return gradient;
}





Eigen::MatrixXd normalRandomMatrix(int rows, int cols, double mean, double stddev) {
    Eigen::MatrixXd mat(rows, cols);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = distribution(generator);
        }
    }
    return mat;
}

Eigen::VectorXd normalRandomVector(int size, double mean, double stddev) {
    Eigen::VectorXd vec(size);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < size; ++i) {
        vec(i) = distribution(generator);
    }
    return vec;
}



Layer::Layer(int input_size, int output_size, std::string activation_func)
        : weights(normalRandomMatrix(output_size, input_size, 0, 0.01)),
        biases(normalRandomVector(output_size, 0, 0.01)),
        activation(std::move(activation_func)),
        finput(Eigen::MatrixXd::Zero(output_size, input_size)),
        weights_gradients(Eigen::MatrixXd::Zero(output_size, input_size)),
        bias_gradients(Eigen::VectorXd::Zero(output_size)) {}

Eigen::MatrixXd Layer::apply_activation(const Eigen::MatrixXd& x) {
        //std::cout << "layer_weight_vector: \n" << x << std::endl;
        if (activation == "relu") {
            return x.cwiseMax(0.0);
        } else if (activation == "sigmoid") {
            return 1.0 / (1.0 + (-x).array().exp());
        } else if (activation == "softmax"){
            return softmax(x);
        } else {
            
            return Eigen::VectorXd::Zero(x.size());
        }
    }

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd& input) {

    Eigen::MatrixXd z = (weights * input).colwise() + biases;

    finput = z;
    return apply_activation(z);
}

void Layer::update_gradients(const double& learning_rate, const int& batch_size) {
    weights_gradients /= batch_size;
    bias_gradients /= batch_size;
    weights -= learning_rate * weights_gradients;
    biases -= learning_rate * bias_gradients;
    weights_gradients.setZero();
    bias_gradients.setZero();
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, const std::vector<std::string>& activations) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], activations[i]);
        }
    }
Eigen::MatrixXd NeuralNetwork::predict(const Eigen::MatrixXd& input) {
        Eigen::MatrixXd output = input;
        output_data.clear();
        for (auto& layer : layers) {
            output_data.emplace_back(output);
            //std::cout << "input:" << output << std::endl;
            
            output = layer.forward(output);
        }

        output_data.emplace_back(output);
        //std::cout << output << std::endl;
        return output;
    }
void NeuralNetwork::backward(const Eigen::VectorXi& target) {
    Eigen::MatrixXd last_layer_output = layers.back().finput;
    Eigen::MatrixXd gradient = softmax_crossentropy_gradient(last_layer_output, target);

    for (size_t i = layers.size() - 1; i > 0; --i) {

        Layer& layer = layers[i];
        Eigen::MatrixXd grad_w = gradient * output_data[i].transpose();
        layer.weights_gradients += grad_w;
        layer.bias_gradients += gradient.rowwise().sum();

        if (i > 0) {
            Eigen::MatrixXd z = layers[i - 1].finput;
            Eigen::MatrixXd d_activ;
            if (layers[i - 1].activation == "relu") {
                d_activ = (z.array() > 0).cast<double>();
            }
            else {
                d_activ = Eigen::MatrixXd::Ones(z.rows(), z.cols());
            }
            gradient = (layer.weights.transpose() * gradient).cwiseProduct(d_activ);
        }
    }
    output_data.clear();
}

void NeuralNetwork::update_weights_and_bias(const double& learning_rate, const int& batch_size) {
    for (auto& layer : layers) {
        layer.update_gradients(learning_rate ,batch_size);
    }
}

void NeuralNetwork::save_weight(const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        for (const auto& layer : layers) {
            int rows = layer.weights.rows();
            int cols = layer.weights.cols();
            file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
            file.write(reinterpret_cast<const char*>(layer.weights.data()), layer.weights.size() * sizeof(double));

            int bias_size = layer.biases.size();
            file.write(reinterpret_cast<const char*>(&bias_size), sizeof(int));
            file.write(reinterpret_cast<const char*>(layer.biases.data()), layer.biases.size() * sizeof(double));
        }
        file.close();
        std::cout << "Weights saved to " << filename << std::endl;
    }
    else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }

}
void NeuralNetwork::load_weight(const std::string& filename) {
    std::vector<Eigen::MatrixXd> matrices;
    std::ifstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        for (auto& layer : layers) {
            int rows, cols, bias_size;
            if(file.read(reinterpret_cast<char*>(&rows), sizeof(int)) &&
                file.read(reinterpret_cast<char*>(&cols), sizeof(int))) {
                Eigen::MatrixXd matrix(rows, cols);
                file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));
                layer.weights.resize(rows, cols);
                layer.weights = matrix;
            }

            if (file.read(reinterpret_cast<char*>(&bias_size), sizeof(int))) {
                Eigen::VectorXd bias(bias_size);
                file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(double));
                layer.biases = bias;
            }
        }
        file.close();
        std::cout << "Weights loaded from " << filename << std::endl;
    }
    else {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }

}
   





