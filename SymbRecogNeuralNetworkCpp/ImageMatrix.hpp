#ifndef IMAGEMATRIX_HPP
#define IMAGEMATRIX_HPP

#include <vector>

namespace SymbRecogNeuralNetworkCpp
{

    struct ImageMatrix
    {
        int Width;
        int Height;
        std::vector<std::vector<bool>> Pixels;

        ImageMatrix(int width, int height, const std::vector<unsigned char>& pixels);
        std::vector<double> ToNormalizedVector() const;
    };

    bool operator<(const ImageMatrix& lhs, const ImageMatrix& rhs);
}


#endif // IMAGEMATRIX_HPPnamespace std