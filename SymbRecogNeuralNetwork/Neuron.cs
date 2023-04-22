using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SymbRecogNeuralNetwork
{
    public class Neuron
    {
        public double[] Weights { get; set; } // веса нейрона
        public double Bias { get; set; } // смещение нейрона

        public double LastActivation { get; private set; } // последнее значение активации нейрона
        public double LastError { get; set; } // последнее значение ошибки при активации нейрона

        public Neuron(int inputCount)
        {
            Weights = new double[inputCount];
            Randomize();
        }

        public double Activate(double[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("Количество входных значений должно быть равно количеству весов нейрона");

            double sum = 0;

            for (int i = 0; i < inputs.Length; i++)
                sum += inputs[i] * Weights[i];

            sum += Bias;

            LastActivation = Sigmoid(sum);

            return LastActivation;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private void Randomize()
        {
            Random random = new Random();

            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = random.NextDouble() * 2 - 1;

            Bias = random.NextDouble() * 2 - 1;
        }
    }

}
