#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <random>
#include <cmath>

#include <nlohmann/json.hpp>

namespace SymbRecogNeuralNetworkCpp
{
    class Neuron
    {
    public:
        std::vector<double> Weights; // веса нейрона
        double Bias; // смещение нейрона

        double LastActivation; // последнее значение активации нейрона
        double LastError; // последнее значение ошибки при активации нейрона

        Neuron(int inputCount);
        double Activate(const std::vector<double>& inputs);

    private:
        double Sigmoid(double x) const;
        void Randomize();
    };
}

namespace nlohmann {
    template <>
    struct adl_serializer<SymbRecogNeuralNetworkCpp::Neuron> {
        static void to_json(json& j, const SymbRecogNeuralNetworkCpp::Neuron& n) {
            j = json{
                {"weights", n.Weights},
                {"bias", n.Bias},
                {"lastActivation", n.LastActivation},
                {"lastError", n.LastError}
            };
        }

        static void from_json(const json& j, SymbRecogNeuralNetworkCpp::Neuron& n) {
            j.at("weights").get_to(n.Weights);
            j.at("bias").get_to(n.Bias);
            j.at("lastActivation").get_to(n.LastActivation);
            j.at("lastError").get_to(n.LastError);
        }
    };
}


#endif // NEURON_HPP