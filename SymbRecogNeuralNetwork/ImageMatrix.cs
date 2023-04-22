using System.Drawing;

namespace SymbRecogNeuralNetwork
{
    public class ImageMatrix
    {
        public int Width { get; }
        public int Height { get; }
        public bool[,] Pixels { get; }

        public ImageMatrix(int width, int height)
        {
            Width = width;
            Height = height;
            Pixels = new bool[width, height];
        }

        public ImageMatrix(bool[,] pixels)
        {
            Width = pixels.GetLength(0);
            Height = pixels.GetLength(1);
            Pixels = pixels;
        }

        public ImageMatrix(string[] lines)
        {
            Width = lines[0].Length;
            Height = lines.Length;
            Pixels = new bool[Width, Height];

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    Pixels[x, y] = lines[y][x] == '1';
                }
            }
        }

        public ImageMatrix(string imageFileName)
        {
            Bitmap bitmap = new Bitmap(imageFileName);
            Width = bitmap.Width;
            Height = bitmap.Height;
            Pixels = new bool[Width, Height];

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    Color color = bitmap.GetPixel(x, y);
                    Pixels[x, y] = color.GetBrightness() < 0.5;
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
