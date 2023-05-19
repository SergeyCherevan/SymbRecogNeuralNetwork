#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

#include "NeuralNetwork.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    NeuralNetwork::NeuralNetwork() { }

    NeuralNetwork::NeuralNetwork(int inputCount, int hiddenCount, int outputCount, int epochs, double learningRate) :
        InputLayer(inputCount, Neuron(1)), // ������������� ������� ��������
        HiddenLayer(hiddenCount, Neuron(inputCount)), // ������������� ������� ��������
        OutputLayer(outputCount, Neuron(hiddenCount)), // ������������� �������� ��������
        Epochs(epochs),
        LearningRate(learningRate) { }

    void NeuralNetwork::Train(std::map<ImageMatrix, std::string> data)
    {
        // �������� ����
        for (int i = 0; i < Epochs; i++)
        {
            std::cout << "���������� ����� " << i << std::endl;

            int j = 0;
            for (auto const& [image, label] : data)
            {
                std::cout << "����� " << i << ". ����������� �" << j << ", ������ \"" << label << "\"" << std::endl;

                // ������ ��������������� �������
                Feedforward(image.ToNormalizedVector());

                // �������� ��������������� ������
                Backpropagate(GetExpectedOutput(label));

                // ���������� ����� ��������
                UpdateWeights();

                j++;
            }

            std::cout << "������������� ����� " << i << std::endl;
        }
    }


    std::string NeuralNetwork::Recognize(ImageMatrix image, double threshold)
    {
        // ������� ������ �������� �������� �����������
        const std::vector<double> inputs = image.ToNormalizedVector();

        // ���������� �������� �������� ����� ����
        Feedforward(inputs);

        // ������� ������ ������� � ���������� ��������� ��������� � �������� ����
        int maxIndex = 0;
        for (int i = 1; i < OutputLayer.size(); i++)
        {
            if (OutputLayer[i].LastActivation > OutputLayer[maxIndex].LastActivation)
            {
                maxIndex = i;
            }
        }

        // ���� �������� ��������� ����������� ������� ������ ���������� ��������, �� �������, ��� ������ �� ���������
        if (OutputLayer[maxIndex].LastActivation < threshold)
        {
            return "Unknown";
        }

        return LabelMapping[maxIndex];
    }

    void NeuralNetwork::SaveToFile(std::string filePath) const
    {
        std::ofstream output(filePath);
        if (output.is_open())
        {
            nlohmann::json jsonNetwork;
            nlohmann::adl_serializer<NeuralNetwork>::to_json(jsonNetwork, *this);


            output << jsonNetwork;
            output.close();
        }
    }

    void NeuralNetwork::ReadFromFile(std::string filePath)
    {
        std::ifstream fileStream(filePath);
        if (!fileStream.is_open()) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        nlohmann::json jsonData;
        fileStream >> jsonData;

        nlohmann::adl_serializer<NeuralNetwork>::from_json(jsonData, *this);
    }

    void NeuralNetwork::Feedforward(const std::vector<double>& inputs)
    {
        // �������� �������� ������� ��������
        for (int i = 0; i < InputLayer.size(); i++)
        {
            InputLayer[i].Activate({ inputs[i] });
        }

        // �������� �������� ������� ��������
        for (int i = 0; i < HiddenLayer.size(); i++)
        {
            HiddenLayer[i].Activate(GetInputs(InputLayer));
        }

        // �������� �������� �������� ��������
        for (int i = 0; i < OutputLayer.size(); i++)
        {
            OutputLayer[i].Activate(GetInputs(HiddenLayer));
        }
    }

    void NeuralNetwork::Backpropagate(const std::vector<double>& expected)
    {
        // ��������� ������ �������� ��������
        for (int i = 0; i < OutputLayer.size(); i++)
        {
            Neuron& outputNeuron = OutputLayer[i];
            double error = expected[i] - outputNeuron.LastActivation;
            outputNeuron.LastError = error * outputNeuron.LastActivation * (1 - outputNeuron.LastActivation);
        }

        // ��������� ������ ������� ��������
        for (int i = 0; i < HiddenLayer.size(); i++) {
            Neuron& hiddenNeuron = HiddenLayer[i];
            double error = 0;
            for (int j = 0; j < OutputLayer.size(); j++) {
                Neuron& outputNeuron = OutputLayer[j];
                error += outputNeuron.LastError * outputNeuron.Weights[i];
            }
            hiddenNeuron.LastError = error * hiddenNeuron.LastActivation * (1 - hiddenNeuron.LastActivation);
        }
    }

    void NeuralNetwork::UpdateWeights()
    {
        // ��������� ���� ������� ��������
        for (size_t i = 0; i < HiddenLayer.size(); i++)
        {
            Neuron& hiddenNeuron = HiddenLayer[i];
            for (size_t j = 0; j < InputLayer.size(); j++)
            {
                Neuron& inputNeuron = InputLayer[j];

                double deltaWeight = LearningRate * hiddenNeuron.LastError * inputNeuron.LastActivation;
                hiddenNeuron.Weights[j] += deltaWeight;
            }
        }

        // ��������� ���� �������� ��������
        for (size_t i = 0; i < OutputLayer.size(); i++)
        {
            Neuron& outputNeuron = OutputLayer[i];
            for (size_t j = 0; j < HiddenLayer.size(); j++)
            {
                Neuron& hiddenNeuron = HiddenLayer[j];

                double deltaWeight = LearningRate * outputNeuron.LastError * hiddenNeuron.LastActivation;
                outputNeuron.Weights[j] += deltaWeight;
            }
        }
    }


    std::vector<double> NeuralNetwork::GetInputs(const std::vector<Neuron>& layer) const
    {
        std::vector<double> inputs(layer.size());

        for (int i = 0; i < layer.size(); i++)
        {
            inputs[i] = layer[i].LastActivation;
        }

        return inputs;
    }

    std::vector<double> NeuralNetwork::GetExpectedOutput(std::string label) const
    {
        int symbolIndex = -1;
        for (const auto& pair : LabelMapping)
        {
            if (pair.second == label)
            {
                symbolIndex = pair.first;
                break;
            }
        }

        std::vector<double> output(OutputLayer.size(), 0.0);
        if (symbolIndex >= 0 && symbolIndex < OutputLayer.size())
        {
            output[symbolIndex] = 1.0;
        }

        return output;
    }
}

