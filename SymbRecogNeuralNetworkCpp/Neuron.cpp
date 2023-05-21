#include "Neuron.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    Neuron::Neuron(int inputCount, std::string neuronType = "output") {
        if (neuronType == "input") {
            Weights = std::vector<double>(1);
            Bias = 0.0;
        }
        else {
            Weights.resize(inputCount);
            Randomize();
        }
    }

    void Neuron::Randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (size_t i = 0; i < Weights.size(); i++)
            Weights[i] = dis(gen);

        Bias = dis(gen);
    }

    double Neuron::Activate(const std::vector<double>& inputs) {
        if (inputs.size() != Weights.size())
            throw std::invalid_argument("Количество входных значений должно быть равно количеству весов нейрона");

        double sum = 0;

        for (size_t i = 0; i < inputs.size(); i++)
            sum += inputs[i] * Weights[i];

        sum += Bias;

        LastActivation = Sigmoid(sum);

        return LastActivation;
    }

    double Neuron::Sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }
}