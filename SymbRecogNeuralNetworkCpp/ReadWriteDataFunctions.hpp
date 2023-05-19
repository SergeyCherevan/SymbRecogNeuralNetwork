#ifndef READWRITEDARAFUNCTIONS_HPP
#define READWRITEDARAFUNCTIONS_HPP

#include <map>
#include <fstream>

#include "ImageMatrix.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    std::map<int, std::string> LoadEmnistLabelMapping(const std::string& mappingFilePath);

    std::map<ImageMatrix, std::string> LoadEmnistData(
        const std::string& imagesFilePath,
        const std::string& labelsFilePath,
        const std::string& mappingFilePath);

    int ReadInt32BigEndian(std::ifstream& reader);
}

#endif // READWRITEDARAFUNCTIONS_HPP
