using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SymbRecogNeuralNetworkCs
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string projectDirectory = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "..");
            Environment.CurrentDirectory = projectDirectory;

            Dictionary<int, string> symbols = LoadEmnistLabelMapping("./Data/emnist-balanced-mapping.txt");

            NeuralNetwork neuralNetwork = new NeuralNetwork();

            Console.WriteLine("На основе чего настраивать нейросеть: датасет изображений (d) или файл с весами нейросети (f)?");
            string doReadTrainingDatasetOrNeuralNetworkWeights = Console.ReadLine();
            if (doReadTrainingDatasetOrNeuralNetworkWeights == "d")
            {
                Dictionary<ImageMatrix, string> trainingData = LoadEmnistData(
                    "./Data/emnist-balanced-train-images-idx3-ubyte",
                    "./Data/emnist-balanced-train-labels-idx1-ubyte",
                    "./Data/emnist-balanced-mapping.txt"
                );

                Console.WriteLine($"Тренировочный датасет содержит {trainingData.Count} изображений для {symbols.Count} символов");

                neuralNetwork = new NeuralNetwork(
                    inputCount: trainingData.First().Key.ToNormalizedArray().Length,
                    hiddenCount: 4,
                    outputCount: symbols.Count,
                    epochs: 20,
                    learningRate: 0.1
                );

                neuralNetwork.LabelMapping = symbols;

                neuralNetwork.Train(trainingData);

                Console.WriteLine("Обучение завершено.");
            }
            else if (doReadTrainingDatasetOrNeuralNetworkWeights == "f")
            {
                Console.WriteLine("Напишите название файла с сохранёнными весами нейросети?");
                string fileName = Console.ReadLine();

                neuralNetwork.ReadFromFile($"./Data/{fileName}");

                neuralNetwork.LabelMapping = symbols;

                Console.WriteLine("Чтение сохранённых весов нейросети завершено.");
            }

            Console.WriteLine("Нажмите любой символ, чтобы начать тестирование модели.");
            Console.ReadKey();

            Dictionary<ImageMatrix, string> testingData = LoadEmnistData(
                "./Data/emnist-balanced-test-images-idx3-ubyte",
                "./Data/emnist-balanced-test-labels-idx1-ubyte",
                "./Data/emnist-balanced-mapping.txt"
            );

            Console.WriteLine($"Тестовый датасет содержит {testingData.Count} изображений для {symbols.Count} символов");

            int j = 0;

            foreach ((ImageMatrix image, string expectedSymbol) in testingData)
            {
                string result = neuralNetwork.Recognize(image, threshold: 0.01);

                Console.WriteLine($"№{j} Распознанный символ: {result}, ожидаемый символ: {expectedSymbol}");

                j++;
            }

            Console.WriteLine("Тестирование завершено.");

            Console.WriteLine("Вы хотите записать веса текущей нейросети в файл (y/n)?");
            string doWriteWeightsToFile = Console.ReadLine();
            if (doWriteWeightsToFile == "y")
            {
                Console.WriteLine("Напишите название файла для сохранения весов нейросети?");
                string fileName = Console.ReadLine();

                neuralNetwork.SaveToFile($"./Data/{fileName}");

                Console.WriteLine("Запись весов нейросети в файл завершена.");
            }

            Console.WriteLine("Нажмите любой символ, чтобы завершить программу.");
            Console.ReadKey();
        }

        public static Dictionary<int, string> LoadEmnistLabelMapping(string mappingFilePath)
        {
            Dictionary<int, string> symbols = new Dictionary<int, string>();

            using (StreamReader reader = new StreamReader(mappingFilePath))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Split(' ');

                    int label = int.Parse(parts[0]);
                    int unicodeValue = int.Parse(parts[1]);

                    char symbol = char.ConvertFromUtf32(unicodeValue)[0];
                    symbols.Add(label, symbol.ToString());
                }
            }

            return symbols;
        }


        public static Dictionary<ImageMatrix, string> LoadEmnistData(string imagesFilePath, string labelsFilePath, string mappingFilePath)
        {
            Dictionary<int, string> labelMapping = LoadEmnistLabelMapping(mappingFilePath);

            Dictionary<ImageMatrix, string> data = new Dictionary<ImageMatrix, string>();

            // Read labels file and corresponding images
            using (BinaryReader imagesReader = new BinaryReader(File.OpenRead(imagesFilePath)))
            using (BinaryReader labelsReader = new BinaryReader(File.OpenRead(labelsFilePath)))
            {
                int magicNumber = ReadInt32BigEndian(imagesReader);
                int numberOfImages = ReadInt32BigEndian(imagesReader);
                int rows = ReadInt32BigEndian(imagesReader);
                int cols = ReadInt32BigEndian(imagesReader);
                int labelMagicNumber = ReadInt32BigEndian(labelsReader);
                int numberOfLabels = ReadInt32BigEndian(labelsReader);

                if (numberOfImages != numberOfLabels)
                {
                    throw new Exception("Number of images and labels does not match");
                }

                for (int i = 0; i < numberOfImages; i++)
                {
                    byte[] pixels = imagesReader.ReadBytes(rows * cols);
                    byte label = labelsReader.ReadByte();

                    string symbol = labelMapping[label];

                    ImageMatrix imageMatrix = new ImageMatrix(rows, cols, pixels);
                    data[imageMatrix] = symbol;
                }
            }

            return data;
        }

        public static int ReadInt32BigEndian(BinaryReader reader)
        {
            byte[] bytes = reader.ReadBytes(4);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
