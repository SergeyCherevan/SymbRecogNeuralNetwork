using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static System.Net.Mime.MediaTypeNames;

namespace SymbRecogNeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var trainingData = ReadDataFromImages("./TrainingData/");

            NeuralNetwork neuralNetwork = new NeuralNetwork(
                inputCount: trainingData.First().Key.ToNormalizedArray().Length,
                hiddenCount: 4,
                outputCount: Enum.GetNames(typeof(SymbolEnum)).Length - 1,
                epochs: 10000,
                learningRate: 0.1
            );

            neuralNetwork.Train(trainingData);

            Console.WriteLine("Обучение завершено.");

            var testingData = ReadDataFromImages("./TestingData/");
            foreach ((ImageMatrix image, SymbolEnum expectedSymbol) in testingData)
            {
                SymbolEnum result = neuralNetwork.Recognize(image, threshold: 0.5);

                Console.WriteLine($"Распознанный символ: {result}, ожидаемый символ: {expectedSymbol}");
            }

            Console.ReadLine();
        }


        private static Dictionary<ImageMatrix, SymbolEnum> ReadDataFromFiles(string dataPath)
        {
            Dictionary<ImageMatrix, SymbolEnum> trainingData = new Dictionary<ImageMatrix, SymbolEnum>();

            string[] fileNames = Directory.GetFiles(dataPath, "*.txt");

            foreach (string fileName in fileNames)
            {
                string[] parts = Path.GetFileNameWithoutExtension(fileName).Split('.');
                SymbolEnum symbol = (SymbolEnum)char.ConvertToUtf32(parts[0], 0);
                int version = int.Parse(parts[1]);

                string[] lines = File.ReadAllLines(fileName).ToArray();
                ImageMatrix matrix = new ImageMatrix(lines);

                trainingData[matrix] = symbol;
            }

            return trainingData;
        }

        private static Dictionary<ImageMatrix, SymbolEnum> ReadDataFromImages(string dataPath)
        {
            Dictionary<ImageMatrix, SymbolEnum> trainingData = new Dictionary<ImageMatrix, SymbolEnum>();

            string[] fileNames = Directory.GetFiles(dataPath, "*.bmp");

            foreach (string fileName in fileNames)
            {
                string[] parts = Path.GetFileNameWithoutExtension(fileName).Split('.');
                SymbolEnum symbol = (SymbolEnum)char.ConvertToUtf32(parts[0], 0);
                int version = int.Parse(parts[1]);

                ImageMatrix matrix = new ImageMatrix(fileName);

                trainingData[matrix] = symbol;
            }

            return trainingData;
        }

    }
}
