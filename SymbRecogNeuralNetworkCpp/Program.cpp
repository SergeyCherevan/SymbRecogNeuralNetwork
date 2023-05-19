// SymbRecogNeuralNetworkCpp.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <locale>

#include "NeuralNetwork.hpp"
#include "ReadWriteDataFunctions.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    int main()
    {
        setlocale(LC_ALL, "Russian");

        std::string projectDirectory = ".."; // указать путь к проекту
        std::filesystem::current_path(projectDirectory); // установить текущую директорию

        std::map<int, std::string> symbols = LoadEmnistLabelMapping("./Data/emnist-balanced-mapping.txt");

        NeuralNetwork neuralNetwork;

        std::cout << "На основе чего настраивать нейросеть: датасет изображений (d) или файл с весами нейросети (f)?" << std::endl;
        std::string doReadTrainingDatasetOrNeuralNetworkWeights;
        std::getline(std::cin, doReadTrainingDatasetOrNeuralNetworkWeights);
        if (doReadTrainingDatasetOrNeuralNetworkWeights == "d")
        {
            std::map<ImageMatrix, std::string> trainingData = LoadEmnistData(
                "./Data/emnist-balanced-train-images-idx3-ubyte",
                "./Data/emnist-balanced-train-labels-idx1-ubyte",
                "./Data/emnist-balanced-mapping.txt"
            );

            std::cout << "Тренировочный датасет содержит " << trainingData.size() << " изображений для " << symbols.size() << " символов" << std::endl;

            neuralNetwork = NeuralNetwork(
                trainingData.begin()->first.ToNormalizedVector().size(),
                20,
                symbols.size(),
                100,
                0.01
            );

            neuralNetwork.LabelMapping = symbols;

            neuralNetwork.Train(trainingData);

            std::cout << "Обучение завершено." << std::endl;
        }
        else if (doReadTrainingDatasetOrNeuralNetworkWeights == "f")
        {
            std::cout << "Напишите название файла с сохранёнными весами нейросети?" << std::endl;
            std::string fileName;
            std::getline(std::cin, fileName);

            neuralNetwork.ReadFromFile("./Data/" + fileName);

            std::cout << "Чтение сохранённых весов нейросети завершено." << std::endl;
        }

        std::cout << "Нажмите любой символ, чтобы начать тестирование модели." << std::endl;
        std::cin.get();

        std::map<ImageMatrix, std::string> testingData = LoadEmnistData(
            "./Data/emnist-balanced-test-images-idx3-ubyte",
            "./Data/emnist-balanced-test-labels-idx1-ubyte",
            "./Data/emnist-balanced-mapping.txt"
        );

        std::cout << "Тестовый датасет содержит " << testingData.size() << " изображений для " << symbols.size() << " символов" << std::endl;

        int j = 0;

        for (const auto& [image, expectedSymbol] : testingData)
        {
            std::string result = neuralNetwork.Recognize(image, 0.01);

            std::cout << "№" << j << " Распознанный символ: " << result << ", ожидаемый символ: " << expectedSymbol << std::endl;

            j++;
        }

        std::cout << "Тестирование завершено." << std::endl;

        std::cout << "Вы хотите записать веса текущей нейросети в файл (y/n)?" << std::endl;
        std::string doWriteWeightsToFile;
        std::cin >> doWriteWeightsToFile;

        if (doWriteWeightsToFile == "y")
        {
            std::cout << "Напишите название файла для сохранения весов нейросети?" << std::endl;
            std::string fileName;
            std::cin >> fileName;

            neuralNetwork.SaveToFile("./Data/" + fileName);

            std::cout << "Запись весов нейросети в файл завершена." << std::endl;
        }

        std::cout << "Нажмите любой символ, чтобы завершить программу." << std::endl;
        std::cin.ignore();
        std::cin.get();

        return 0;
    }
}

int main()
{
    try
    {
        SymbRecogNeuralNetworkCpp::main();
    }
    catch (std::exception exp)
    {
        std::cerr << typeid(exp).name() << ": " << exp.what();
    }
}