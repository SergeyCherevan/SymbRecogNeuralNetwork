#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <map>
#include <string>
#include "Neuron.hpp"
#include "ImageMatrix.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    class NeuralNetwork
    {
    public:
        std::map<int, std::string> LabelMapping;
        int Epochs;
        double LearningRate;

        std::vector<Neuron> InputLayer;
        std::vector<Neuron> HiddenLayer;
        std::vector<Neuron> OutputLayer;

        NeuralNetwork();
        NeuralNetwork(int inputCount, int hiddenCount, int outputCount, int epochs, double learningRate);

        void Train(std::map<ImageMatrix, std::string> data);
        std::string Recognize(ImageMatrix image, double threshold);

        void SaveToFile(std::string filePath) const;
        void ReadFromFile(std::string filePath);

        std::string serializeToJson() const;

    private:
        void Feedforward(const std::vector<double>& inputs);
        std::vector<double> GetInputs(const std::vector<Neuron>& layer) const;
        void Backpropagate(const std::vector<double>& expected);
        std::vector<double> GetExpectedOutput(std::string label) const;
        void UpdateWeights();
    };
}

namespace nlohmann {
    template <>
    struct adl_serializer<SymbRecogNeuralNetworkCpp::NeuralNetwork> {
        static void to_json(json& j, const SymbRecogNeuralNetworkCpp::NeuralNetwork& nn)
        {
            // add network parameters to JSON object
            j["inputCount"] = nn.InputLayer.size();
            j["hiddenCount"] = nn.HiddenLayer.size();
            j["outputCount"] = nn.OutputLayer.size();
            j["epochs"] = nn.Epochs;
            j["learningRate"] = nn.LearningRate;

            // add label mapping to JSON object
            j["labelMapping"] = nn.LabelMapping;

            // add network layers to JSON object
            for (auto neuron : nn.InputLayer)
            {
                json jsonNeuron;
                adl_serializer<SymbRecogNeuralNetworkCpp::Neuron>::to_json(jsonNeuron, neuron);
                j["inputLayer"].push_back(jsonNeuron);
            }
            for (auto neuron : nn.HiddenLayer)
            {
                json jsonNeuron;
                adl_serializer<SymbRecogNeuralNetworkCpp::Neuron>::to_json(jsonNeuron, neuron);
                j["hiddenLayer"].push_back(jsonNeuron);
            }
            for (auto neuron : nn.OutputLayer)
            {
                json jsonNeuron;
                adl_serializer<SymbRecogNeuralNetworkCpp::Neuron>::to_json(jsonNeuron, neuron);
                j["outputLayer"].push_back(jsonNeuron);
            }
        }

        static void from_json(const json& j, SymbRecogNeuralNetworkCpp::NeuralNetwork& nn)
        {
            nn.LabelMapping = j["labelMapping"].get<std::map<int, std::string>>();
            nn.Epochs = j["epochs"].get<int>();
            nn.LearningRate = j["learningRate"].get<double>();

            nn.InputLayer.clear();
            nn.HiddenLayer.clear();
            nn.OutputLayer.clear();

            auto inputLayerJson = j["inputLayer"];
            for (const auto& neuronJson : inputLayerJson)
            {
                SymbRecogNeuralNetworkCpp::Neuron n(1);
                adl_serializer<SymbRecogNeuralNetworkCpp::Neuron>::from_json(neuronJson, n);

                nn.InputLayer.push_back(n);
            }

            auto hiddenLayerJson = j["hiddenLayer"];
            for (const auto& neuronJson : hiddenLayerJson)
            {
                SymbRecogNeuralNetworkCpp::Neuron n(nn.InputLayer.size());
                adl_serializer<SymbRecogNeuralNetworkCpp::Neuron>::from_json(neuronJson, n);

                nn.HiddenLayer.push_back(n);
            }

            auto outputLayerJson = j["outputLayer"];
            for (const auto& neuronJson : outputLayerJson)
            {
                SymbRecogNeuralNetworkCpp::Neuron n(nn.HiddenLayer.size());
                adl_serializer<SymbRecogNeuralNetworkCpp::Neuron>::from_json(neuronJson, n);

                nn.OutputLayer.push_back(n);
            }
        }
    };
}

#endif // NEURALNETWORK_HPP