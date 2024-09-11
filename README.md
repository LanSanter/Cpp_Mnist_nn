
# Neural Network for MNIST Classification
This project implements a simple feedforward neural network to classify handwritten digits from the MNIST dataset. It uses Eigen for matrix operations and zlib for reading compressed MNIST data.

## Features
- **Custom Neural Network**: A neural network with fully connected layers is implemented from scratch, supporting different activation functions such as ReLU and Softmax.
- **MNIST Data Loader**: Supports loading and preprocessing of the MNIST dataset (images and labels).
- **Forward Propagation**: Predict outputs for given inputs.
- **Backpropagation**: Update weights and biases based on cross-entropy loss and gradient descent.
- **Weight Saving/Loading**: Save trained weights to a file and load them for future use.
- **Accuracy Calculation**: Calculates the accuracy of predictions.
## Requirements
- **C++ Compiler**: Support for C++11 or higher.
- **Eigen**: Linear algebra library for matrix and vector operations. You can download it from [Eigen Official Website](https://eigen.tuxfamily.org/dox/GettingStarted.html).
- **zlib**: Used for reading compressed MNIST dataset files. You can download it from [zlib Official Website](https://zlib.net/).

Please download and extract these libraries to a project directory named `eigen-3.4.0` and `zlib-1.3.1` respectively.

## Project Structure
```markdown
- Neuron_layer.h
- Neuron_layer.cpp
- mnist_reader.h
- mnist_reader.cpp
- main.cpp
- mnist (directory containing MNIST dataset)
    - train-images-idx3-ubyte.gz
    - train-labels-idx1-ubyte.gz
    ...
```
### File Descriptions
- **Neuron_layer.h / Neuron_layer.cpp**: Contains the definition and implementation of the `Layer` and NeuralNetwork classes, which form the structure of the neural network.
- **mnist_reader.h / mnist_reader.cpp**: Handles reading MNIST image and label files.
- **main.cpp**: The main entry point of the program where the MNIST dataset is loaded, the neural network is trained, and results are printed.
## How to Use
### 1. Clone the Repository
You can clone this repository to your local machine:

```bash
git clone https://github.com/username/repository-name.git
```
### 2. Install Dependencies
Make sure to download and include **Eigen** and **zlib** in your project:

- **Eigen**: Place the Eigen library in a directory called eigen-3.4.0/.
- **zlib**: Place the zlib library in a directory called zlib-1.3.1/.
Alternatively, link them using your project configuration (e.g., in Visual Studio or Makefile).

### 3. Download MNIST Dataset
Download the MNIST dataset files and place them in a mnist/ directory

### 4. Compile the Program (Using Visual Studio)
1. Open **Visual Studio**.
2. Go to **File > Open > Project/Solution** and select your `.sln` file (e.g., `NeuronNetwork.sln`).
3. Ensure that the `eigen` and `zlib` directories are included in your project, or properly configure the **Include Directories** and **Library Directories**:
 - Right-click the project name (e.g., `NeuronNetwork`) in **Solution Explorer** and select **Properties**.
 - Under **VC++ Directories**, set the **Include Directories** to point to `eigen-3.4.0` and `zlib-1.3.1`.
 - Under **Linker > General**, set **Additional Library Directories** to point to the `zlib` library directory (e.g., `zlib-1.3.1/lib`).
4. Under **Linker > Input**, ensure that **Additional Dependencies** includes `zlib.lib` or the appropriate zlib library file.
5. Click **Build > Build Solution** (or use the shortcut `Ctrl+Shift+B`) to compile the project.
### 5. Run the Program (Using Visual Studio)
1. Ensure that your project is set to either **Debug** or **Release** mode (you can select this from the top right).
2. Click **Debug > Start Without Debugging** or press `Ctrl+F5` to run the program.
After running, the program will load the MNIST dataset, train the neural network, and output accuracy and related information to the console.

### 6. Training Output
The program will train the neural network for a number of epochs on the MNIST dataset and output the accuracy at regular intervals. The trained weights will be saved to a file called `weights.bin`.

## Example Usage
```cpp
int epoch = 2;  // Set the number of training epochs
double learning_rate = 0.3;  // Set the learning rate

// Initialize neural network with 3 layers
std::vector<int> layer_size = {784, 1000, 10};
std::vector<std::string> activations = {"relu", "softmax"};

NeuralNetwork net(layer_size, activations); 
net.load_weight("weights.bin");  // Load pre-trained weights

// Train network using MNIST data
for (int j = 0; j < epoch; ++j) {
    // Forward propagation, backpropagation, and weight updates happen here
    net.update_weights_and_bias(learning_rate, batch_size);
}

net.save_weight("weights.bin");  // Save trained weights
```

## License
This project is licensed under the MIT License 

