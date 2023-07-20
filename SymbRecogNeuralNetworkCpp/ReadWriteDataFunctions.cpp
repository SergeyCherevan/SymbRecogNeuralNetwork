#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

#include "ReadWriteDataFunctions.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    std::unordered_map<int, std::string> LoadEmnistLabelMapping(const std::string& mappingFilePath)
    {
        std::unordered_map<int, std::string> symbols;

        std::ifstream input(mappingFilePath);
        if (!input.is_open())
        {
            throw std::runtime_error("Failed to open file: " + mappingFilePath);
        }

        int label;
        int unicodeValue;
        while (input >> label >> unicodeValue)
        {
            char symbol = static_cast<char>(unicodeValue);
            symbols.emplace(label, std::string(1, symbol));
        }

        return symbols;
    }

    // Функция для загрузки данных из файлов EMNIST
    std::unordered_map<ImageMatrix, std::string> LoadEmnistData(
        const std::string& imagesFilePath,
        const std::string& labelsFilePath,
        const std::string& mappingFilePath)
    {
        std::unordered_map<int, std::string> labelMapping = LoadEmnistLabelMapping(mappingFilePath);

        std::unordered_map<ImageMatrix, std::string> data;
        data.reserve(labelMapping.size()); // Предварительное выделение памяти

        // Чтение файлов изображений и меток
        std::ifstream imagesReader(imagesFilePath, std::ios::binary);
        std::ifstream labelsReader(labelsFilePath, std::ios::binary);

        if (!imagesReader || !labelsReader)
        {
            throw std::runtime_error("Unable to open input files");
        }

        int magicNumber = ReadInt32BigEndian(imagesReader);
        int numberOfImages = ReadInt32BigEndian(imagesReader);
        int rows = ReadInt32BigEndian(imagesReader);
        int cols = ReadInt32BigEndian(imagesReader);
        int labelMagicNumber = ReadInt32BigEndian(labelsReader);
        int numberOfLabels = ReadInt32BigEndian(labelsReader);

        if (numberOfImages != numberOfLabels)
        {
            throw std::runtime_error("Number of images and labels does not match");
        }

        for (int i = 0; i < numberOfImages; i++)
        {
            std::vector<unsigned char> pixels(rows * cols);
            imagesReader.read(reinterpret_cast<char*>(pixels.data()), rows * cols);
            unsigned char label = labelsReader.get();

            std::string symbol = labelMapping[static_cast<int>(label)];

            ImageMatrix imageMatrix(rows, cols, pixels);
            data.emplace(imageMatrix, std::move(symbol));
        }

        return data;
    }

    // Вспомогательная функция для чтения 4-байтового числа в формате big-endian
    int ReadInt32BigEndian(std::ifstream& reader)
    {
        char buffer[4];
        reader.read(buffer, 4);
        return (static_cast<unsigned char>(buffer[0]) << 24) |
            (static_cast<unsigned char>(buffer[1]) << 16) |
            (static_cast<unsigned char>(buffer[2]) << 8) |
            static_cast<unsigned char>(buffer[3]);
    }
}