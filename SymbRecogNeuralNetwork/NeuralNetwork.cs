﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SymbRecogNeuralNetwork
{
    public class NeuralNetwork
    {
        public Neuron[] InputLayer { get; private set; }
        public Neuron[] HiddenLayer { get; private set; }
        public Neuron[] OutputLayer { get; private set; }

        public int Epochs { get; private set; }
        public double LearningRate { get; private set; }



        public NeuralNetwork(int inputCount, int hiddenCount, int outputCount, int epochs, double learningRate)
        {
            InputLayer = new Neuron[inputCount];
            HiddenLayer = new Neuron[hiddenCount];
            OutputLayer = new Neuron[outputCount];

            // инициализация входных нейронов
            for (int i = 0; i < inputCount; i++)
            {
                InputLayer[i] = new Neuron(1); // входные нейроны имеют один вход
            }

            // инициализация скрытых нейронов
            for (int i = 0; i < hiddenCount; i++)
            {
                HiddenLayer[i] = new Neuron(inputCount);
            }

            // инициализация выходных нейронов
            for (int i = 0; i < outputCount; i++)
            {
                OutputLayer[i] = new Neuron(hiddenCount);
            }

            Epochs = epochs;
            LearningRate = learningRate;
        }

        public void Train(Dictionary<ImageMatrix, SymbolEnum> data)
        {
            // обучение сети
            for (int i = 0; i < Epochs; i++)
            {
                Console.WriteLine($"Начинается эпоха {i}");

                foreach (var item in data)
                {
                    // прямое распространение сигнала
                    Feedforward(item.Key.ToNormalizedArray());

                    // обратное распространение ошибки
                    Backpropagate(GetExpectedOutput(item.Value));

                    // обновление весов нейронов
                    UpdateWeights();
                }

                Console.WriteLine($"Заканчивается эпоха {i}");
            }
        }

        public SymbolEnum Recognize(ImageMatrix image, double threshold)
        {
            // создаем массив значений пикселей изображения
            double[] inputs = image.ToNormalizedArray();

            // пропускаем значения пикселей через сеть
            Feedforward(inputs);

            // находим индекс нейрона с наибольшим значением активации в выходном слое
            int maxIndex = 0;
            for (int i = 1; i < OutputLayer.Length; i++)
            {
                if (OutputLayer[i].LastActivation > OutputLayer[maxIndex].LastActivation)
                {
                    maxIndex = i;
                }
            }

            // если значение активации наибольшего нейрона меньше порогового значения, то считаем, что символ не распознан
            if (OutputLayer[maxIndex].LastActivation < threshold)
            {
                return SymbolEnum.Unknown;
            }

            SymbolEnum[] values = Enum.GetValues<SymbolEnum>();

            return values[maxIndex];

        }



        public void Feedforward(double[] inputs)
        {
            // передаем значения входных нейронов
            for (int i = 0; i < InputLayer.Length; i++)
            {
                InputLayer[i].Activate(new double[] { inputs[i] });
            }

            // передаем значения скрытым нейронам
            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                HiddenLayer[i].Activate(GetInputs(InputLayer));
            }

            // передаем значения выходным нейронам
            for (int i = 0; i < OutputLayer.Length; i++)
            {
                OutputLayer[i].Activate(GetInputs(HiddenLayer));
            }
        }

        private void Backpropagate(double[] expected)
        {
            // вычисляем ошибку выходных нейронов
            for (int i = 0; i < OutputLayer.Length; i++)
            {
                Neuron outputNeuron = OutputLayer[i];
                double error = expected[i] - outputNeuron.LastActivation;
                outputNeuron.LastError = error * outputNeuron.LastActivation * (1 - outputNeuron.LastActivation);
            }

            // вычисляем ошибку скрытых нейронов
            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                Neuron hiddenNeuron = HiddenLayer[i];
                double error = 0;
                for (int j = 0; j < OutputLayer.Length; j++)
                {
                    Neuron outputNeuron = OutputLayer[j];
                    error += outputNeuron.LastError * outputNeuron.Weights[i];
                }
                hiddenNeuron.LastError = error * hiddenNeuron.LastActivation * (1 - hiddenNeuron.LastActivation);
            }
        }

        private void UpdateWeights()
        {
            // обновляем веса скрытых нейронов
            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                Neuron hiddenNeuron = HiddenLayer[i];
                for (int j = 0; j < InputLayer.Length; j++)
                {
                    Neuron inputNeuron = InputLayer[j];
                    double deltaWeight = LearningRate * hiddenNeuron.LastError * inputNeuron.LastActivation;
                    hiddenNeuron.Weights[j] += deltaWeight;
                }
            }

            // обновляем веса выходных нейронов
            for (int i = 0; i < OutputLayer.Length; i++)
            {
                Neuron outputNeuron = OutputLayer[i];
                for (int j = 0; j < HiddenLayer.Length; j++)
                {
                    Neuron hiddenNeuron = HiddenLayer[j];
                    double deltaWeight = LearningRate * outputNeuron.LastError * hiddenNeuron.LastActivation;
                    outputNeuron.Weights[j] += deltaWeight;
                }
            }
        }




        private double[] GetInputs(Neuron[] layer)
        {
            double[] inputs = new double[layer.Length];

            for (int i = 0; i < layer.Length; i++)
            {
                inputs[i] = layer[i].LastActivation;
            }

            return inputs;
        }

        private double[] GetExpectedOutput(SymbolEnum symbol)
        {
            SymbolEnum[] values = Enum.GetValues<SymbolEnum>();
            int symbolIndex = Array.IndexOf(values, symbol);

            double[] output = new double[OutputLayer.Length];
            output[symbolIndex] = 1;

            return output;
        }

    }
}
