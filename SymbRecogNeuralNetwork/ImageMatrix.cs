using System.Drawing;

namespace SymbRecogNeuralNetwork
{
    public class ImageMatrix
    {
        public int Width { get; }
        public int Height { get; }
        public bool[,] Pixels { get; }

        public ImageMatrix(int width, int height, byte[] pixels)
        {
            Width = width;
            Height = height;
            Pixels = new bool[Width, Height];

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    Pixels[x, y] = pixels[y * Width + x] > 127;
                }
            }
        }

        public double[] ToNormalizedArray()
        {
            double[] result = new double[Width * Height];
            int index = 0;

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    result[index++] = Pixels[x, y] ? 1.0 : 0.0;
                }
            }

            return result;
        }
    }

}
