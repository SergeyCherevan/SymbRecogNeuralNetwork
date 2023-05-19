#include <vector>
#include "ImageMatrix.hpp"

namespace SymbRecogNeuralNetworkCpp
{
    
    ImageMatrix::ImageMatrix(int width, int height, const std::vector<unsigned char>& pixels)
        : Width(width), Height(height), Pixels(height, std::vector<bool>(width)) {

        for (int y = 0; y < Height; y++) {
            for (int x = 0; x < Width; x++) {
                Pixels[y][x] = pixels[y * Width + x] > 127;
            }
        }
    }

    std::vector<double> ImageMatrix::ToNormalizedVector() const
    {
        std::vector<double> result(Width * Height);
        int index = 0;

        for (int y = 0; y < Height; y++) {
            for (int x = 0; x < Width; x++) {
                result[index++] = Pixels[y][x] ? 1.0 : 0.0;
            }
        }

        return result;
    }

    bool operator<(const ImageMatrix& lhs, const ImageMatrix& rhs)
    {
        if (lhs.Width != rhs.Width) {
            return lhs.Width < rhs.Width;
        }
        if (lhs.Height != rhs.Height) {
            return lhs.Height < rhs.Height;
        }
        const auto& lhsPixels = lhs.Pixels;
        const auto& rhsPixels = rhs.Pixels;
        for (size_t i = 0; i < lhsPixels.size(); ++i) {
            const auto& lhsRow = lhsPixels[i];
            const auto& rhsRow = rhsPixels[i];
            if (lhsRow != rhsRow) {
                return lhsRow < rhsRow;
            }
        }
        return false;
    }
}
