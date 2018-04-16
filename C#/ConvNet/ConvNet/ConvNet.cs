using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.IO;

namespace ConvNet
{

    public enum LayerType
    {
        Convolutional, Pool, FullConnected, Hidden
    }

    public enum CNNType
    {
        CNN, NN
    }

    public enum PoolType
    {
        Max, Avg
    }

    public enum ColorModel
    {
        RGB, Grayscale, HSV
    }

    public enum ActivationFcn
    {
        Tanh, Sigm, TanhConv, ReLU, TanhLinear
    }

    public class Dataset
    {
        int OrigW;
        int OrigH;
        int ImageWidth;
        int ImageHeight;
        ColorModel ColorMod;
        public int ClassCount { private set; get; }
        public double[][][,] Data { private set; get; }//image count; channel count; array of pixels.       
        public int[] Answers { private set; get; }//выходные классы
        public double[][] AnswersR { private set; get; }//выходные значения для задач регрессии
        public Bitmap[] Originals { private set; get; }//изображения для возможности показать их при обучении
        public Bitmap[] ClassExamples { private set; get; }// примеры изображений для каждого класса
        public double[] Mins { private set; get; }// минимальные значения до нормирования
        public double[] Maxes { private set; get; }// максимальные значения до нормирования
        List<double[][,]> TempData;
        List<int> TempAnswers;
        List<Bitmap> TempOriginals;
        List<double[]> TempAnswersR;

        /// <summary>
        /// Создает новую выборку. Если данные  не изображения, используется только количество классов
        /// </summary>
        /// <param name="ClassCount">количество классов</param>
        /// <param name="ColorMod">цветовая модель</param>
        /// <param name="ImageWidth">ширина изображения</param>
        /// <param name="ImageHeight">высота изображения</param>
        public Dataset(int ClassCount, ColorModel ColorMod, int ImageWidth, int ImageHeight)
        {
            ClassExamples = new Bitmap[ClassCount];
            this.ImageHeight = ImageHeight;
            this.ImageWidth = ImageWidth;
            this.ColorMod = ColorMod;
            this.ClassCount = ClassCount;
            this.TempAnswers = new List<int>();
            this.TempAnswersR = new List<double[]>();
            this.TempData = new List<double[][,]>();
            this.TempOriginals = new List<Bitmap>();
            OrigW = 200;
            OrigH = 200;
        }
        /// <summary>
        /// добавляет экземпляр и его класс
        /// </summary>
        /// <param name="data"></param>
        /// <param name="Answer"></param>
        public void AddData(double[] data, int Answer)
        {
            TempAnswers.Add(Answer);
            double[][,] temp = new double[1][,];
            int count = data.Count();
            temp[0] = new double[count, 1];
            for (int i = 0; i < count; i++)
                temp[0][i, 0] = data[i];
            TempData.Add(temp);
        }
        /// <summary>
        /// добавляет экземпляр и выходные значения
        /// </summary>
        /// <param name="data"></param>
        /// <param name="Answers"></param>
        public void AddData(double[] data, double[] Answers)
        {
            double[] ans = new double[Answers.Count()];
            for (int i = 0; i < Answers.Count(); i++)
                ans[i] = Answers[i];
            TempAnswersR.Add(ans);
            double[][,] temp = new double[1][,];
            int count = data.Count();
            temp[0] = new double[count, 1];
            for (int i = 0; i < count; i++)
                temp[0][i, 0] = data[i];
            TempData.Add(temp);
        }

        /// <summary>
        /// добавляет 3-мерные данные и выходные значения (экспериментальная функция)
        /// </summary>
        /// <param name="data"></param>
        /// <param name="Answers"></param>
        public void Add3dData(double[,,] data, double[] Answers)
        {
            double[] ans = new double[Answers.Count()];
            for (int i = 0; i < Answers.Count(); i++)
                ans[i] = Answers[i];
            TempAnswersR.Add(ans);
            int d0 = data.GetLength(0);
            int d1 = data.GetLength(1);
            int d2 = data.GetLength(2);
            double[][,] temp = new double[data.GetLength(0)][,];
            for (int i = 0; i < d0; i++)
            {
                temp[i] = new double[d1, d2];
                for (int j = 0; j < d1; j++)
                    for (int k = 0; k < d2; k++)
                        temp[i][j, k] = data[i, j, k];
            }
            TempData.Add(temp);
        }

        /// <summary>
        /// добавляет изображение
        /// </summary>
        /// <param name="Path">путь к иображению</param>
        /// <param name="Answer">класс</param>
        /// <param name="FlipVert">отразить по вертикали</param>
        /// <param name="FlipHor">отразить по горизонтали</param>
        /// <param name="FlipAll">отразить по горизонтали и вертикали</param>
        public void AddImage(string Path, int Answer, bool FlipVert, bool FlipHor, bool FlipAll)
        {
            Image img = Image.FromFile(Path);
            Bitmap bmp = new Bitmap(img, ImageWidth, ImageHeight);
            TempAnswers.Add(Answer);
            if (ColorMod == ColorModel.RGB)
                TempData.Add(ToRGB(bmp));
            if (ColorMod == ColorModel.Grayscale)
                TempData.Add(ToGrayscale(bmp));
            if (ColorMod == ColorModel.HSV)
                TempData.Add(ToHSV(bmp));
            TempOriginals.Add(new Bitmap(img, OrigW, OrigH));
            if (FlipVert)
            {
                img.RotateFlip(RotateFlipType.RotateNoneFlipY);
                bmp = new Bitmap(img, ImageWidth, ImageHeight);
                TempAnswers.Add(Answer);
                if (ColorMod == ColorModel.RGB)
                    TempData.Add(ToRGB(bmp));
                if (ColorMod == ColorModel.Grayscale)
                    TempData.Add(ToGrayscale(bmp));
                if (ColorMod == ColorModel.HSV)
                    TempData.Add(ToHSV(bmp));
                TempOriginals.Add(new Bitmap(img, OrigW, OrigH));
                img.RotateFlip(RotateFlipType.RotateNoneFlipY);
            }
            if (FlipHor)
            {
                img.RotateFlip(RotateFlipType.RotateNoneFlipX);
                bmp = new Bitmap(img, ImageWidth, ImageHeight);
                TempAnswers.Add(Answer);
                if (ColorMod == ColorModel.RGB)
                    TempData.Add(ToRGB(bmp));
                if (ColorMod == ColorModel.Grayscale)
                    TempData.Add(ToGrayscale(bmp));
                if (ColorMod == ColorModel.HSV)
                    TempData.Add(ToHSV(bmp));
                TempOriginals.Add(new Bitmap(img, OrigW, OrigH));
                img.RotateFlip(RotateFlipType.RotateNoneFlipX);
            }
            if (FlipAll)
            {
                img.RotateFlip(RotateFlipType.RotateNoneFlipXY);
                bmp = new Bitmap(img, ImageWidth, ImageHeight);
                TempAnswers.Add(Answer);
                if (ColorMod == ColorModel.RGB)
                    TempData.Add(ToRGB(bmp));
                if (ColorMod == ColorModel.Grayscale)
                    TempData.Add(ToGrayscale(bmp));
                if (ColorMod == ColorModel.HSV)
                    TempData.Add(ToHSV(bmp));
                TempOriginals.Add(new Bitmap(img, OrigW, OrigH));
            }
            bmp.Dispose();
            img.Dispose();
        }


        double[][,] ToRGB(Bitmap bmp)//преобразовать изображение в массив RGB
        {
            double[][,] res = new double[3][,];
            for (int i = 0; i < 3; i++)
                res[i] = new double[ImageWidth, ImageHeight];
            Color color;
            for (int i = 0; i < ImageWidth; i++)
                for (int j = 0; j < ImageHeight; j++)
                {
                    color = bmp.GetPixel(i, j);
                    res[0][i, j] = (double)color.R / 255.0;
                    res[1][i, j] = (double)color.G / 255.0;
                    res[2][i, j] = (double)color.B / 255.0;
                }
            return res;
        }

        double[][,] ToGrayscale(Bitmap bmp)//преобразовать изображение в массив Grayscale
        {
            double[][,] res = new double[1][,];
            res[0] = new double[bmp.Width, bmp.Height];
            Color color;
            double Gray;
            for (int i = 0; i < bmp.Width; i++)
                for (int j = 0; j < bmp.Height; j++)
                {
                    color = bmp.GetPixel(i, j);
                    Gray = color.R * 0.3 + color.G * 0.59 + color.B * 0.1;
                    res[0][i, j] = Gray / 255.0;///255
                }

            return res;
        }

        double[][,] ToHSV(Bitmap bmp)//преобразовать изображение в массив HSV
        {
            double[][,] res = new double[3][,];
            for (int i = 0; i < 3; i++)
                res[i] = new double[ImageWidth, ImageHeight];
            double min, max;
            Color color;
            for (int i = 0; i < ImageWidth; i++)
                for (int j = 0; j < ImageHeight; j++)
                {
                    color = bmp.GetPixel(i, j);
                    max = color.R;
                    min = color.R;
                    if (max < color.G) max = color.G;
                    if (max < color.B) max = color.B;
                    if (min > color.G) min = color.G;
                    if (min > color.B) min = color.B;
                    if (max == min)
                        res[0][i, j] = 0;
                    else
                    {
                        if (max == color.R && color.G >= color.B)
                            res[0][i, j] = 60.0 * (color.G - color.B) / (max - min);
                        if (max == color.R && color.G < color.B)
                            res[0][i, j] = 60.0 * (color.G - color.B) / (max - min) + 360.0;
                        if (max == color.G)
                            res[0][i, j] = 60.0 * (color.B - color.R) / (max - min) + 120.0;
                        if (max == color.B)
                            res[0][i, j] = 60.0 * (color.R - color.G) / (max - min) + 240.0;
                    }
                    res[0][i, j] /= 360.0;
                    if (max == 0.0)
                        res[1][i, j] = 0;
                    else
                        res[1][i, j] = min / max;
                    res[2][i, j] = max / 255.0;
                }
            return res;
        }

        /// <summary>
        /// подготовить выборку к обучению
        /// </summary>
        public void PrepareData()
        {
            if (TempAnswers.Count != 0)
                Answers = TempAnswers.ToArray();
            else
                AnswersR = TempAnswersR.ToArray();
            Originals = TempOriginals.ToArray();
            Data = TempData.ToArray();
            TempData.Clear();
            TempAnswers.Clear();
            TempAnswersR.Clear();
            TempOriginals.Clear();
        }
        /// <summary>
        /// нормирование выборки
        /// </summary>
        public void Normalize()
        {
            int ExCount = Data.Count();
            int ParamCount = Data[0][0].GetLength(0);
            double max;
            double min;

            for (int i = 0; i < ParamCount; i++)
            {
                max = double.MinValue;
                min = double.MaxValue;
                for (int j = 0; j < ExCount; j++)
                {
                    if (Data[j][0][i, 0] < min) min = Data[j][0][i, 0];
                    if (Data[j][0][i, 0] > max) max = Data[j][0][i, 0];
                }
                for (int j = 0; j < ExCount; j++)
                {
                    if (min != max)
                    {
                        Data[j][0][i, 0] -= min;
                        Data[j][0][i, 0] /= (max - min);
                    }
                    else
                        Data[j][0][i, 0] = 0;
                }
            }

            if (AnswersR != null)
            {
                ParamCount = AnswersR[0].Count();
                Mins = new double[ParamCount];
                Maxes = new double[ParamCount];
                for (int i = 0; i < ParamCount; i++)
                {
                    max = double.MinValue;
                    min = double.MaxValue;
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (AnswersR[j][i] < min) min = AnswersR[j][i];
                        if (AnswersR[j][i] > max) max = AnswersR[j][i];
                    }
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (min != max)
                        {
                            AnswersR[j][i] -= min;
                            AnswersR[j][i] /= (max - min);
                        }
                        else
                            AnswersR[j][i] = 0;
                    }
                    Mins[i] = min;
                    Maxes[i] = max;
                }
            }

        }

        /// <summary>
        /// заполняет пропущенные значения средними арифметическими. Пропущенные значения должны быть равны -1 (экспериментальная функция)
        /// </summary>
        public void FillMissing()
        {
            int ExCount = Data.Count();
            int ParamCount = Data[0][0].GetLength(0);
            double sum;
            int cnt;
            for (int i = 0; i < ParamCount; i++)
            {
                sum = 0;
                cnt = 0;
                for (int j = 0; j < ExCount; j++)
                {
                    if (Data[j][0][i, 0] != -1) sum += Data[j][0][i, 0];
                    cnt++;
                }
                sum /= cnt;
                for (int j = 0; j < ExCount; j++)
                {
                    if (Data[j][0][i, 0] == -1) Data[j][0][i, 0] = sum;

                }
            }
            if (AnswersR != null)
            {
                ParamCount = AnswersR[0].Count();
                for (int i = 0; i < ParamCount; i++)
                {
                    sum = 0;
                    cnt = 0;
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (AnswersR[j][i] != -1) sum += AnswersR[j][i];
                        cnt++;
                    }
                    sum /= cnt;
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (AnswersR[j][i] == -1) AnswersR[j][i] = sum;

                    }
                }
            }
        }

        /// <summary>
        /// исправляет выбросы в данных (экспериментальтая функция)
        /// </summary>
        /// <param name="MinBord"></param>
        /// <param name="MaxBord"></param>
        /// <param name="RemoveBord"></param>
        public void RemoveOutlires(double MinBord, double MaxBord, double RemoveBord)
        {
            int ExCount = Data.Count();
            int ParamCount = Data[0][0].GetLength(0);
            double max;
            double min;
            int countSm;
            int countMd;
            int countBg;
            for (int i = 0; i < ParamCount; i++)
            {
                max = double.MinValue;
                min = double.MaxValue;
                countSm = 0;
                countMd = 0;
                countBg = 0;
                for (int j = 0; j < ExCount; j++)
                {
                    if (Data[j][0][i, 0] < min) min = Data[j][0][i, 0];
                    if (Data[j][0][i, 0] > max) max = Data[j][0][i, 0];
                }
                for (int j = 0; j < ExCount; j++)
                {
                    if (Data[j][0][i, 0] <= min + MinBord * (max - min))
                        countSm++;
                    if (Data[j][0][i, 0] >= min + MaxBord * (max - min))
                        countBg++;
                    if (Data[j][0][i, 0] > min + MinBord * (max - min) && Data[j][0][i, 0] < min + MaxBord * (max - min))
                        countMd++;
                }
                if ((double)countMd / (double)(countMd + countBg + countSm) > RemoveBord)
                {
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (Data[j][0][i, 0] <= min + MinBord * (max - min))
                            Data[j][0][i, 0] = min + MinBord * (max - min);
                        if (Data[j][0][i, 0] >= min + MaxBord * (max - min))
                            Data[j][0][i, 0] = min + MaxBord * (max - min);
                    }
                    continue;
                }
                if ((double)countSm / (double)(countMd + countBg + countSm) > RemoveBord)
                {
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (Data[j][0][i, 0] >= min + MinBord * (max - min))
                            Data[j][0][i, 0] = min + MinBord * (max - min);
                    }
                    continue;
                }
                if ((double)countBg / (double)(countMd + countBg + countSm) > RemoveBord)
                {
                    for (int j = 0; j < ExCount; j++)
                    {
                        if (Data[j][0][i, 0] <= min + MaxBord * (max - min))
                            Data[j][0][i, 0] = min + MaxBord * (max - min);
                    }
                    continue;
                }

            }

        }

        /// <summary>
        /// выделяет контуры в изображениях (экспериментальная функция)
        /// </summary>
        public void TransformToOutlines()
        {
            int Count = Data.Count();
            int Channels = Data[0].Count();
            int W = Data[0][0].GetLength(0);
            int H = Data[0][0].GetLength(1);
            Parallel.For(0, Count, i =>
            {
                double[][,] temp = new double[Channels][,];
                int cnt = 0; ;
                for (int j = 0; j < Channels; j++)
                {
                    temp[j] = new double[W, H];
                    for (int w = 0; w < W; w++)
                        for (int h = 0; h < H; h++)
                        {
                            temp[j][w, h] = 0;
                            cnt = 0;
                            if (w != 0)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w - 1, h]) * 2.0;
                                cnt += 2;
                            }
                            if (h != 0)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w, h - 1]) * 2.0;
                                cnt += 2;
                            }
                            if (w != W - 1)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w + 1, h]) * 2.0;
                                cnt += 2;
                            }
                            if (h != H - 1)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w, h + 1]) * 2.0;
                                cnt += 2;
                            }
                            if (w != 0 && h != 0)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w - 1, h - 1]);
                                cnt++;
                            }
                            if (w != 0 && h != H - 1)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w - 1, h + 1]);
                                cnt++;
                            }
                            if (w != W - 1 && h != 0)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w + 1, h - 1]);
                                cnt++;
                            }
                            if (w != W - 1 && h != H - 1)
                            {
                                temp[j][w, h] += Math.Abs(Data[i][j][w, h] - Data[i][j][w + 1, h + 1]);
                                cnt++;
                            }

                            temp[j][w, h] /= (double)cnt;
                            temp[j][w, h] *= 1.5;
                            if (temp[j][w, h] > 2) temp[j][w, h] = 2.0;
                            temp[j][w, h] -= 1.0;
                        }

                }
                for (int j = 0; j < Channels; j++)
                    for (int w = 0; w < W; w++)
                        for (int h = 0; h < H; h++)
                        {
                            Data[i][j][w, h] = temp[j][w, h];
                        }
            });


        }

        /// <summary>
        /// перемешивает всю выборку
        /// </summary>
        public void MixData()
        {
            MixData(Data.Count());
        }

        /// <summary>
        /// перемешивает выборку
        /// </summary>
        /// <param name="count">количество экземпляров в начале выборки, которые нужно перемешать</param>
        public void MixData(int count)
        {
            Random rand = new Random();
            for (int i = 0; i < count; i++)
            {
                int swapIndex = rand.Next(count - i) + i;
                double[][,] tempData = Data[i];
                Data[i] = Data[swapIndex];
                Data[swapIndex] = tempData;
                if (Answers != null)
                {
                    int tempAnswers = Answers[i];
                    Answers[i] = Answers[swapIndex];
                    Answers[swapIndex] = tempAnswers;
                }
                else
                {
                    double[] tempAnswers = AnswersR[i];
                    AnswersR[i] = AnswersR[swapIndex];
                    AnswersR[swapIndex] = tempAnswers;
                }
                if (Originals.Count() != 0)
                {
                    Bitmap bmp = Originals[i];
                    Originals[i] = Originals[swapIndex];
                    Originals[swapIndex] = bmp;
                }

            }
        }

        /// <summary>
        /// загружает изображения, которые можно показывать в качестве примера класса при обучении
        /// </summary>
        /// <param name="path"></param>
        /// <param name="Class"></param>
        public void LoadClassExample(string path, int Class)
        {
            ClassExamples[Class] = new Bitmap(Image.FromFile(path), 200, 200);
        }

    }

    public class Layer
    {
        public LayerType Type { private set; get; }
        public ActivationFcn ActFcn { private set; get; }
        public int KernelCount { private set; get; }
        public int KernelWidth { private set; get; }
        public int KernelHeight { private set; get; }
        public int StepRight { private set; get; }
        public int StepDown { private set; get; }
        public bool CombineChannels { private set; get; }
        public int PoolWidth { private set; get; }
        public int PoolHeight { private set; get; }
        public PoolType PoolT { private set; get; }
        public int HiddenCount { private set; get; }

        /// <summary>
        /// создает сверточный слой
        /// </summary>
        /// <param name="KernelCount">количество ядер свертки</param>
        /// <param name="KernelWidth">ширина ядра</param>
        /// <param name="KernelHeight">высота ядра</param>
        /// <param name="StepRight">сдвиг по горизонтали</param>
        /// <param name="StepDown">сдвиг по вертикали</param>
        /// <param name="CombineChannels">объединить цветовые каналы в один на выходе</param>
        /// <param name="ActFcn">функция активации</param>
        public Layer(int KernelCount, int KernelWidth, int KernelHeight, int StepRight, int StepDown, bool CombineChannels, ActivationFcn ActFcn)
        {
            this.ActFcn = ActFcn;
            this.Type = LayerType.Convolutional;
            this.KernelCount = KernelCount;
            this.KernelWidth = KernelWidth;
            this.KernelHeight = KernelHeight;
            this.StepRight = StepRight;
            this.StepDown = StepDown;
            this.CombineChannels = CombineChannels;
        }

        /// <summary>
        /// создает слой субдискретизации
        /// </summary>
        /// <param name="PoolWidth">ширина окна</param>
        /// <param name="PoolHeight">высота окна</param>
        /// <param name="PoolT">тип субдискретизации</param>
        /// <param name="KernelCount">количество ядер свертки предыдущего слоя</param>
        public Layer(int PoolWidth, int PoolHeight, PoolType PoolT, int KernelCount)
        {
            this.KernelCount = KernelCount;
            this.Type = LayerType.Pool;
            this.PoolHeight = PoolHeight;
            this.PoolWidth = PoolWidth;
            this.PoolT = PoolT;
        }

        /// <summary>
        /// создает скрытый слой
        /// </summary>
        /// <param name="NeuronsCount">количество выходных нейронов</param>
        /// <param name="Type">тип слоя</param>
        /// <param name="ActFcn">функция активации</param>
        public Layer(int NeuronsCount, LayerType Type, ActivationFcn ActFcn)
        {
            this.ActFcn = ActFcn;
            this.Type = Type;
            this.HiddenCount = NeuronsCount;
            if (Type != LayerType.Hidden && Type != LayerType.FullConnected)
                throw new Exception("Invalid layer type! Only hidden or full connected.");
        }

    }

    public class Kernel
    {
        public double[][][][,] Weights; //kernel count; channel count; array of weights.
        public double[,] Bias;

        public Kernel(Layer layer, int Channels, int PrewCount)//создает ядра свертки или набор весов по параметрам слоя
        {
            if (layer.Type == LayerType.Convolutional)
            {
                Weights = new double[PrewCount][][][,];
                for (int p = 0; p < PrewCount; p++)
                {
                    Weights[p] = new double[layer.KernelCount][][,];
                    for (int i = 0; i < layer.KernelCount; i++)
                    {
                        Weights[p][i] = new double[Channels][,];
                        for (int j = 0; j < Channels; j++)
                        {
                            Weights[p][i][j] = new double[layer.KernelWidth, layer.KernelHeight];
                        }
                    }
                }
                Bias = new double[layer.KernelCount, Channels];
                if (layer.CombineChannels)
                    Bias = new double[layer.KernelCount, 1];
            }
            else
            {
                Weights = new double[1][][][,];
                Weights[0] = new double[1][][,];
                Weights[0][0] = new double[1][,];
                Weights[0][0][0] = new double[layer.HiddenCount, Channels];
                Bias = new double[layer.HiddenCount, 1];
            }
        }

        public Kernel(Layer layer, int Width, int Height, int KernelCount)//создает ядра свертки или набор весов по параметрам слоя
        {
            Weights = new double[1][][][,];
            Weights[0] = new double[layer.HiddenCount][][,];
            for (int i = 0; i < layer.HiddenCount; i++)
            {
                Weights[0][i] = new double[KernelCount][,];
                for (int j = 0; j < KernelCount; j++)
                    Weights[0][i][j] = new double[Width, Height];
            }
            Bias = new double[layer.HiddenCount, 1];
        }

        public void SetRandomWeights()//инициализирует веса случайными значениями
        {
            Random rnd = new Random();
            Parallel.For(0, Weights.Count(), p =>
            {
                for (int i = 0; i < Weights[p].Count(); i++)
                    for (int j = 0; j < Weights[p][i].Count(); j++)
                        for (int k = 0; k < Weights[p][i][j].GetLength(0); k++)
                            for (int n = 0; n < Weights[p][i][j].GetLength(1); n++)
                            {
                                Weights[p][i][j][k, n] = rnd.NextDouble() * 0.2 - 0.1;
                            }
            });
            for (int k = 0; k < Bias.GetLength(0); k++)
                for (int n = 0; n < Bias.GetLength(1); n++)
                    Bias[k, n] = 0;

        }
    }

    public class Out//результат итерации обучения для возможности отображения хода обучения
    {
        public double Cost;
        public bool RecognRes;
        public Out()
        {
            Cost = 0;
        }
    }

    public class CNN
    {
        List<Layer> Layers;
        List<Kernel> Kernels;
        Dataset dataset;
        public CNNType CNNtype { get; private set; }
        public int Iterations;//количество итераций обучения
        public int Epohs;//количество эпох обучения
        public double LearnRate;//скорость обучения
        public double LearnRateDecrease;//коэфициент уменьшения скорости обучения

        /// <summary>
        /// создает новую нейросеть
        /// </summary>
        /// <param name="type">тип</param>
        public CNN(CNNType type)
        {
            this.CNNtype = type;
            Iterations = 0;
            Epohs = 0;
            LearnRate = 0.01;
            Layers = new List<Layer>();
            Kernels = new List<Kernel>();
        }

        /// <summary>
        /// загружает нейросеть из файла
        /// </summary>
        /// <param name="path"></param>
        public CNN(string path)
        {
            Layers = new List<Layer>();
            Kernels = new List<Kernel>();
            LoadCNN(path);
            LearnRate = 0.01;
        }

        /// <summary>
        /// подготавливает нейросеть к обучению по загруженной выборке, проверяет правильность построения нейросети
        /// </summary>
        public void CreateNewCNN()
        {
            int ImgWidth = dataset.Data[0][0].GetLength(0);
            int ImgHeight = dataset.Data[0][0].GetLength(1);
            int ChannelCount = dataset.Data[0].Count();
            bool FullConnExist = false;
            if (CNNtype == CNNType.NN)
                FullConnExist = true;
            int PrewCount = 0;
            for (int i = 0; i < Layers.Count; i++)
            {
                switch (Layers[i].Type)
                {
                    case LayerType.Convolutional:
                        if (FullConnExist)
                            throw new Exception("Unable to create convolutional layer after fully connected layer!");
                        ImgWidth = (ImgWidth - Layers[i].KernelWidth) / Layers[i].StepRight + 1;
                        ImgHeight = (ImgHeight - Layers[i].KernelHeight) / Layers[i].StepDown + 1;
                        if (i != 0)
                            Kernels.Add(new Kernel(Layers[i], ChannelCount, PrewCount));
                        else
                            Kernels.Add(new Kernel(Layers[i], ChannelCount, 1));
                        if (Layers[i].CombineChannels && ChannelCount == 1)
                            throw new Exception("Unable to create convolutional layer with CombineChannels property! Number of channels must be more than one!");
                        if (Layers[i].CombineChannels) ChannelCount = 1;
                        PrewCount = Layers[i].KernelCount;
                        break;
                    case LayerType.FullConnected:
                        if (FullConnExist)
                            throw new Exception("Unable to create more than 1 fully connected layer!");
                        if (ImgHeight < 1 || ImgWidth < 1)
                            throw new Exception("Unable to create fully connected layer! Input images too small!");
                        if (ChannelCount != 1)
                            throw new Exception("Unable to create fully connected layer! Number of channels still more than one!");
                        Kernels.Add(new Kernel(Layers[i], ImgWidth, ImgHeight, Kernels[Kernels.Count - 1].Weights[0].Count()));
                        FullConnExist = true;
                        break;
                    case LayerType.Hidden:
                        if (!FullConnExist)
                            throw new Exception("Unable to create hidden layer! Fully connected layer is not exists!");
                        if (i != 0)
                            Kernels.Add(new Kernel(Layers[i], Layers[i - 1].HiddenCount, 1));
                        else
                            Kernels.Add(new Kernel(Layers[i], dataset.Data[0][0].GetLength(0), 1));
                        break;
                    case LayerType.Pool:
                        if (FullConnExist)
                            throw new Exception("Unable to create pool layer after fully connected layer!");
                        ImgWidth /= Layers[i].PoolWidth;
                        ImgHeight /= Layers[i].PoolHeight;
                        break;
                }
            }
            for (int i = 0; i < Kernels.Count; i++)
                Kernels[i].SetRandomWeights();
        }

        /// <summary>
        /// итерация распознавания данных
        /// </summary>
        /// <param name="Input"></param>
        /// <returns></returns>
        public double[] Run(double[][,] Input)
        {
            double[][][,] Output = new double[1][][,];
            Output[0] = Input;
            int KernelIndex = 0;
            for (int i = 0; i < Layers.Count; i++)
                switch (Layers[i].Type)
                {
                    case LayerType.Pool:
                        Output = Pool(Layers[i], Output);
                        break;
                    case LayerType.Hidden:
                        Output = Hidden(Layers[i], Kernels[KernelIndex], Output);
                        KernelIndex++;
                        break;
                    case LayerType.FullConnected:
                        Output = FullConnect(Layers[i], Kernels[KernelIndex], Output);
                        KernelIndex++;
                        break;
                    case LayerType.Convolutional:
                        Output = Convolution(Layers[i], Kernels[KernelIndex], Output);
                        KernelIndex++;
                        break;
                }
            return OutputToArray(Output);
        }

        public void MixDataset()
        {
            dataset.MixData();
        }

        public void MixDataset(int c)
        {
            dataset.MixData(c);
        }

        /// <summary>
        /// итерация обучения
        /// </summary>
        /// <param name="Pos">индекс экземпляра выборки для обучения</param>
        /// <returns></returns>
        public Out BackPropagate(int Pos)
        {
            Out res = new Out();
            double[][][][,] Output = new double[Layers.Count + 1][][][,];
            Output[0] = new double[1][][,];
            Output[0][0] = dataset.Data[Pos];
            int KernelIndex = 0;
            for (int i = 0; i < Layers.Count; i++)//прямое распространение
                switch (Layers[i].Type)
                {
                    case LayerType.Pool:
                        Output[i + 1] = Pool(Layers[i], Output[i]);
                        break;
                    case LayerType.Hidden:
                        Output[i + 1] = Hidden(Layers[i], Kernels[KernelIndex], Output[i]);
                        KernelIndex++;
                        break;
                    case LayerType.FullConnected:
                        Output[i + 1] = FullConnect(Layers[i], Kernels[KernelIndex], Output[i]);
                        KernelIndex++;
                        break;
                    case LayerType.Convolutional:
                        Output[i + 1] = Convolution(Layers[i], Kernels[KernelIndex], Output[i]);
                        KernelIndex++;
                        break;
                }
            double[][][][,] Gradient = new double[Layers.Count][][][,];
            int ChanIn = 1;
            int ChanOut = 1;
            KernelIndex--;

            double[] DesiredOuts = new double[dataset.ClassCount];
            if (dataset.Answers != null)//определить тип задачи (классификация или регрессия)
            {
                for (int i = 0; i < dataset.ClassCount; i++)//желаемые значения на выходах в задачах классификации зависят от функции активации
                {
                    if (Layers[Layers.Count - 1].ActFcn == ActivationFcn.Sigm)
                        DesiredOuts[i] = 0;
                    if (Layers[Layers.Count - 1].ActFcn == ActivationFcn.Tanh || Layers[Layers.Count - 1].ActFcn == ActivationFcn.TanhLinear)
                        DesiredOuts[i] = -1;
                    if (Layers[Layers.Count - 1].ActFcn == ActivationFcn.TanhConv)
                        DesiredOuts[i] = -1.7159;
                    if (Layers[Layers.Count - 1].ActFcn == ActivationFcn.ReLU)
                        DesiredOuts[i] = -0.01;
                }
                if (Layers[Layers.Count - 1].ActFcn == ActivationFcn.Sigm || Layers[Layers.Count - 1].ActFcn == ActivationFcn.Tanh || Layers[Layers.Count - 1].ActFcn == ActivationFcn.ReLU || Layers[Layers.Count - 1].ActFcn == ActivationFcn.TanhLinear)
                    DesiredOuts[dataset.Answers[Pos]] = 1;
                if (Layers[Layers.Count - 1].ActFcn == ActivationFcn.TanhConv)
                    DesiredOuts[dataset.Answers[Pos]] = 1.7159;
                for (int i = 0; i < dataset.ClassCount; i++)
                    res.Cost += (DesiredOuts[i] - Output[Layers.Count][0][0][i, 0]) * (DesiredOuts[i] - Output[Layers.Count][0][0][i, 0]);
                res.Cost = Math.Sqrt(res.Cost);
                if (OutputToArray(Output[Layers.Count]).Max() == Output[Layers.Count][0][0][dataset.Answers[Pos], 0])//проверяет, был ли текущий экземпляр распознан правильно
                    res.RecognRes = true;
                else
                    res.RecognRes = false;
            }
            else
            {
                DesiredOuts = dataset.AnswersR[Pos];
                for (int i = 0; i < dataset.ClassCount; i++)
                    res.Cost += (DesiredOuts[i] - Output[Layers.Count][0][0][i, 0]) * (DesiredOuts[i] - Output[Layers.Count][0][0][i, 0]);
                res.Cost = Math.Sqrt(res.Cost);
            }
            Gradient[Layers.Count - 1] = BackOutput(Layers[Layers.Count - 1], Output[Layers.Count], DesiredOuts);

            for (int i = Layers.Count - 2; i >= 0; i--)//вычисление градиентов (обратное распространение)
            {
                switch (Layers[i + 1].Type)
                {
                    case LayerType.Convolutional:
                        if (Layers[i + 1].CombineChannels)
                            ChanIn = dataset.Data[Pos].Count();
                        Gradient[i] = BackConvolution(Layers[i], Layers[i + 1], Kernels[KernelIndex], Output[i + 1], Gradient[i + 1], ChanIn, ChanOut);
                        if (Layers[i + 1].CombineChannels)
                            ChanOut = dataset.Data[Pos].Count();
                        KernelIndex--;
                        break;
                    case LayerType.FullConnected:
                        Gradient[i] = BackFullConnect(Layers[i], Layers[i + 1], Kernels[KernelIndex], Output[i + 1], Gradient[i + 1]);
                        KernelIndex--;
                        break;
                    case LayerType.Hidden:
                        Gradient[i] = BackHidden(Layers[i], Layers[i + 1], Kernels[KernelIndex], Output[i + 1], Gradient[i + 1]);
                        KernelIndex--;
                        break;
                    case LayerType.Pool:
                        Gradient[i] = BackPool(Layers[i], Layers[i + 1], Output[i + 1], Output[i + 2], Gradient[i + 1]);
                        break;
                }
            }
            KernelIndex = 0;
            int P = 1;
            ChanIn = 1;
            ChanOut = 1;
            int W;
            int H;
            for (int ln = 0; ln < Layers.Count; ln++)//коррекция весов
            {
                switch (Layers[ln].Type)
                {
                    case LayerType.Convolutional:
                        W = Gradient[ln][0][0].GetLength(0);
                        H = Gradient[ln][0][0].GetLength(1);
                        int Chan1 = Kernels[KernelIndex].Weights[0][0].Count();
                        int Chan2 = Gradient[ln][0].Count();
                        if (ln != 0)
                            P = Layers[ln - 1].KernelCount;
                        if (Gradient[ln][0].Count() == 1)
                            Parallel.For(0, P, a =>
                            {
                                for (int i = 0; i < Layers[ln].KernelCount; i++)
                                    for (int j = 0; j < Chan1; j++)
                                        for (int k = 0; k < Layers[ln].KernelWidth; k++)
                                            for (int m = 0; m < Layers[ln].KernelHeight; m++)
                                                for (int n = 0; n < W; n++)
                                                    for (int l = 0; l < H; l++)
                                                        for (int b = 0; b < Chan2; b++)
                                                            Kernels[KernelIndex].Weights[a][i][j][k, m] -= Output[ln][a][j][n + k, l + m] * Gradient[ln][i][b][n, l] * LearnRate;

                            });
                        if (Gradient[ln][0].Count() != 1)
                            Parallel.For(0, P, a =>
                            {
                                for (int i = 0; i < Layers[ln].KernelCount; i++)
                                    for (int k = 0; k < Layers[ln].KernelWidth; k++)
                                        for (int m = 0; m < Layers[ln].KernelHeight; m++)
                                            for (int n = 0; n < W; n++)
                                                for (int l = 0; l < H; l++)
                                                    for (int b = 0; b < Chan2; b++)
                                                        Kernels[KernelIndex].Weights[a][i][b][k, m] -= Output[ln][a][b][n + k, l + m] * Gradient[ln][i][b][n, l] * LearnRate;

                            });
                        int KCount = Kernels[KernelIndex].Bias.GetLength(0);
                        int BChan = Kernels[KernelIndex].Bias.GetLength(1);
                        Parallel.For(0, KCount, i =>
                        {
                            for (int j = 0; j < BChan; j++)
                                for (int k = 0; k < W; k++)
                                    for (int m = 0; m < H; m++)
                                    {
                                        Kernels[KernelIndex].Bias[i, j] -= Gradient[ln][i][j][k, m] * LearnRate;
                                    }
                        });

                        KernelIndex++;
                        break;
                    case LayerType.FullConnected:
                        W = Output[ln][0][0].GetLength(0);
                        H = Output[ln][0][0].GetLength(1);
                        Parallel.For(0, Layers[ln - 1].KernelCount, i =>
                        {
                            for (int j = 0; j < Layers[ln].HiddenCount; j++)
                                for (int n = 0; n < W; n++)
                                    for (int l = 0; l < H; l++)
                                        Kernels[KernelIndex].Weights[0][j][i][n, l] -= Output[ln][i][0][n, l] * Gradient[ln][0][0][j, 0] * LearnRate;
                        });
                        Parallel.For(0, Layers[ln].HiddenCount, i =>
                        {
                            Kernels[KernelIndex].Bias[i, 0] -= Gradient[ln][0][0][i, 0] * LearnRate;
                        });
                        KernelIndex++;
                        break;
                    case LayerType.Hidden:
                        int PrCount = 0;
                        if (ln != 0)
                            PrCount = Layers[ln - 1].HiddenCount;
                        else
                            PrCount = dataset.Data[0][0].GetLength(0);
                        Parallel.For(0, PrCount, i =>
                        {
                            for (int j = 0; j < Layers[ln].HiddenCount; j++)
                                Kernels[KernelIndex].Weights[0][0][0][j, i] -= Output[ln][0][0][i, 0] * Gradient[ln][0][0][j, 0] * LearnRate;

                        });
                        Parallel.For(0, Layers[ln].HiddenCount, i =>
                        {
                            Kernels[KernelIndex].Bias[i, 0] -= Gradient[ln][0][0][i, 0] * LearnRate;
                        });
                        KernelIndex++;
                        break;
                    case LayerType.Pool:
                        KernelIndex += 0;
                        break;
                }
            }
           
            return res;
        }


        double[][][,] BackOutput(Layer layer, double[][][,] Output, double[] Desired)//вычисление градиентов на выходном слое
        {
            double[][][,] Grad = new double[1][][,];
            Grad[0] = new double[1][,];
            Grad[0][0] = new double[layer.HiddenCount, 1];
            Parallel.For(0, layer.HiddenCount, i =>
            {
                Grad[0][0][i, 0] = (Output[0][0][i, 0] - Desired[i]) * derivActivation(Output[0][0][i, 0], layer.ActFcn);
            });
            return Grad;
        }

        double[][][,] BackHidden(Layer layer, Layer NextLayer, Kernel kernel, double[][][,] Output, double[][][,] Grad)//вычисление градиентов на скрытом слое
        {
            double[][][,] Gradient = new double[1][][,];
            Gradient[0] = new double[1][,];
            Gradient[0][0] = new double[layer.HiddenCount, 1];
            Parallel.For(0, layer.HiddenCount, i =>
            {
                Gradient[0][0][i, 0] = 0;
                for (int j = 0; j < NextLayer.HiddenCount; j++)
                    Gradient[0][0][i, 0] += Grad[0][0][j, 0] * kernel.Weights[0][0][0][j, i] * derivActivation(Output[0][0][i, 0], layer.ActFcn);
            });
            return Gradient;
        }

        double[][][,] BackFullConnect(Layer layer, Layer NextLayer, Kernel kernel, double[][][,] Output, double[][][,] Grad)//вычисление градиентов на слое перехода от карт признаков к одномерным слоям
        {
            double[][][,] Gradient = new double[layer.KernelCount][][,];
            int Width = Output[0][0].GetLength(0);//ширина карт признаков
            int Height = Output[0][0].GetLength(1);//высота карт признаков
            Parallel.For(0, layer.KernelCount, i =>
            {
                Gradient[i] = new double[1][,];
                Gradient[i][0] = new double[Width, Height];
                for (int j = 0; j < Width; j++)
                    for (int k = 0; k < Height; k++)
                        Gradient[i][0][j, k] = 0;
            });
            Parallel.For(0, layer.KernelCount, i =>
            {
                for (int j = 0; j < Width; j++)
                    for (int k = 0; k < Height; k++)
                        for (int m = 0; m < NextLayer.HiddenCount; m++)
                        {
                            switch (layer.Type)
                            {
                                case LayerType.Pool:
                                    Gradient[i][0][j, k] += Grad[0][0][m, 0] * kernel.Weights[0][m][i][j, k];
                                    break;
                                case LayerType.Convolutional:
                                    Gradient[i][0][j, k] += Grad[0][0][m, 0] * kernel.Weights[0][m][i][j, k] * derivActivation(Output[i][0][j, k], layer.ActFcn);
                                    break;
                            }
                        }
            });
            return Gradient;
        }

        double[][][,] BackConvolution(Layer layer, Layer NextLayer, Kernel kernel, double[][][,] Output, double[][][,] Grad, int ChannelsIn, int ChannelsOut)//вычисление градиентов на сверточном слое
        {
            double[][][,] Gradient = new double[layer.KernelCount][][,];
            int Width = Output[0][0].GetLength(0);
            int Height = Output[0][0].GetLength(1);
            Parallel.For(0, layer.KernelCount, i =>
            {
                Gradient[i] = new double[ChannelsIn][,];
                for (int c = 0; c < ChannelsIn; c++)
                {
                    Gradient[i][c] = new double[Width, Height];
                    for (int j = 0; j < Width; j++)
                        for (int k = 0; k < Height; k++)
                            Gradient[i][c][j, k] = 0;
                }
            });
            int NextWidth = Grad[0][0].GetLength(0);
            int NextHeight = Grad[0][0].GetLength(1);
            Parallel.For(0, NextLayer.KernelCount, i =>//количество карт признаков выходного слоя
            {
                for (int j = 0; j < ChannelsOut; j++)//количество цветовых каналов выходного слоя
                    for (int k = 0; k < NextWidth; k++)//ширина карт признаков выходного слоя
                        for (int m = 0; m < NextHeight; m++)//высота карт признаков выходного слоя
                            for (int a = 0; a < layer.KernelCount; a++)// количество карт признаков входного слоя
                                for (int b = 0; b < ChannelsIn; b++)//количество цветовых каналов входного слоя
                                    for (int c = 0; c < NextLayer.KernelWidth; c++)//ширина карт признаков входного слоя
                                        for (int d = 0; d < NextLayer.KernelHeight; d++)//высота карт признаков входного слоя
                                        {
                                            switch (layer.Type)
                                            {
                                                case LayerType.Pool:
                                                    if (ChannelsOut == 1)
                                                        Gradient[a][b][k * NextLayer.StepRight + c, m * NextLayer.StepDown + d] += kernel.Weights[a][i][b][c, d] * Grad[i][j][k, m];
                                                    if (ChannelsOut != 1 && b == 0)
                                                        Gradient[a][j][k * NextLayer.StepRight + c, m * NextLayer.StepDown + d] += kernel.Weights[a][i][j][c, d] * Grad[i][j][k, m];
                                                    break;
                                                case LayerType.Convolutional:

                                                    if (ChannelsOut == 1)
                                                        Gradient[a][b][k * NextLayer.StepRight + c, m * NextLayer.StepDown + d] += kernel.Weights[a][i][b][c, d] * Grad[i][j][k, m] * derivActivation(Output[a][b][k * NextLayer.StepRight + c, m * NextLayer.StepDown + d], layer.ActFcn);
                                                    if (ChannelsOut != 1 && b == 0)
                                                        Gradient[a][j][k * NextLayer.StepRight + c, m * NextLayer.StepDown + d] += kernel.Weights[a][i][j][c, d] * Grad[i][j][k, m] * derivActivation(Output[a][b][k * NextLayer.StepRight + c, m * NextLayer.StepDown + d], layer.ActFcn);
                                                    break;
                                            }
                                        }
            });
            return Gradient;
        }

        double[][][,] BackPool(Layer layer, Layer NextLayer, double[][][,] Output, double[][][,] OutputNext, double[][][,] Grad)//вычисление градиентов на слое субдискретизации
        {
            double[][][,] Gradient = new double[Output.Count()][][,];
            int KernelCount = Output.Count();
            int ChannelCount = Output[0].Count();
            int Width = Output[0][0].GetLength(0);
            int Height = Output[0][0].GetLength(1);
            Parallel.For(0, KernelCount, i =>
            {
                Gradient[i] = new double[ChannelCount][,];
                for (int c = 0; c < ChannelCount; c++)
                {
                    Gradient[i][c] = new double[Width, Height];
                    for (int j = 0; j < Width; j++)
                        for (int k = 0; k < Height; k++)
                            Gradient[i][c][j, k] = 0;
                }
            });
            int WidthNext = OutputNext[0][0].GetLength(0);
            int HeightNext = OutputNext[0][0].GetLength(1);
            Parallel.For(0, KernelCount, i =>
            {
                for (int j = 0; j < ChannelCount; j++)
                    for (int k = 0; k < WidthNext; k++)
                        for (int m = 0; m < HeightNext; m++)
                        {
                            bool IsFind = false;
                            for (int a = 0; a < NextLayer.PoolWidth; a++)
                                for (int b = 0; b < NextLayer.PoolHeight; b++)
                                {
                                    switch (NextLayer.PoolT)
                                    {
                                        case PoolType.Avg:
                                            Gradient[i][j][k * NextLayer.PoolWidth + a, m * NextLayer.PoolHeight + b] = Grad[i][j][k, m] / (NextLayer.PoolWidth * NextLayer.PoolHeight);
                                            break;
                                        case PoolType.Max:
                                            if (Output[i][j][k * NextLayer.PoolWidth + a, m * NextLayer.PoolHeight + b] == OutputNext[i][j][k, m] && !IsFind)
                                            {
                                                IsFind = true;
                                                Gradient[i][j][k * NextLayer.PoolWidth + a, m * NextLayer.PoolHeight + b] = Grad[i][j][k, m];
                                            }
                                            break;
                                    }
                                }
                        }
            });
            if (layer.Type == LayerType.Convolutional)
                Parallel.For(0, KernelCount, i =>
                {
                    for (int j = 0; j < ChannelCount; j++)
                        for (int k = 0; k < Width; k++)
                            for (int m = 0; m < Height; m++)
                                Gradient[i][j][k, m] *= derivActivation(Output[i][j][k, m], layer.ActFcn);

                });
            return Gradient;
        }

        /// <summary>
        /// загрузить выборку данных
        /// </summary>
        /// <param name="dataset"></param>
        public void LoadData(Dataset dataset)
        {
            this.dataset = dataset;
            double Count = dataset.Data.Count();
            if (Count < 50) Count = 50;
            LearnRateDecrease = 1.0 - 0.1 * (1 - Math.Pow(8, -Count / 1000.0));
        }

        /// <summary>
        /// добавить слой
        /// </summary>
        /// <param name="layer"></param>
        public void AddLayer(Layer layer)
        {
            Layers.Add(layer);
        }

        /// <summary>
        /// загрузить нейросеть из файла
        /// </summary>
        /// <param name="path"></param>
        void LoadCNN(string path)
        {
            Layers.Clear();
            Kernels.Clear();
            StreamReader reader = new StreamReader(path);
            Iterations = int.Parse(reader.ReadLine());
            CNNtype = (CNNType)Enum.Parse(typeof(CNNType), reader.ReadLine());
            int pc = 0;
            if (CNNtype == CNNType.NN)
                pc = int.Parse(reader.ReadLine());
            Layer layer;
            Kernel kernel;
            int LC = int.Parse(reader.ReadLine());
            int P = 1;
            for (int i = 0; i < LC; i++)
            {
                switch ((LayerType)Enum.Parse(typeof(LayerType), reader.ReadLine()))
                {
                    case LayerType.Convolutional:
                        layer = new Layer(int.Parse(reader.ReadLine()), int.Parse(reader.ReadLine()), int.Parse(reader.ReadLine()), int.Parse(reader.ReadLine()), int.Parse(reader.ReadLine()), bool.Parse(reader.ReadLine()), (ActivationFcn)Enum.Parse(typeof(ActivationFcn), reader.ReadLine()));
                        Layers.Add(layer);
                        if (i != 0)
                            kernel = new Kernel(layer, int.Parse(reader.ReadLine()), Layers[i - 1].KernelCount);
                        else
                            kernel = new Kernel(layer, int.Parse(reader.ReadLine()), 1);
                        for (int k = 0; k < kernel.Bias.GetLength(0); k++)
                            for (int n = 0; n < kernel.Bias.GetLength(1); n++)
                                kernel.Bias[k, n] = double.Parse(reader.ReadLine());
                        if (i != 0) P = Layers[i - 1].KernelCount;
                        for (int p = 0; p < P; p++)
                            for (int j = 0; j < layer.KernelCount; j++)
                                for (int k = 0; k < kernel.Weights[p][j].Count(); k++)
                                    for (int n = 0; n < layer.KernelWidth; n++)
                                        for (int m = 0; m < layer.KernelHeight; m++)
                                            kernel.Weights[p][j][k][n, m] = double.Parse(reader.ReadLine());
                        Kernels.Add(kernel);
                        break;
                    case LayerType.FullConnected:
                        layer = new Layer(int.Parse(reader.ReadLine()), LayerType.FullConnected, (ActivationFcn)Enum.Parse(typeof(ActivationFcn), reader.ReadLine()));
                        Layers.Add(layer);
                        kernel = new Kernel(layer, int.Parse(reader.ReadLine()), int.Parse(reader.ReadLine()), Kernels[Kernels.Count - 1].Weights[0].Count());
                        for (int j = 0; j < kernel.Bias.GetLength(0); j++)
                            kernel.Bias[j, 0] = double.Parse(reader.ReadLine());
                        for (int n = 0; n < layer.HiddenCount; n++)
                            for (int m = 0; m < kernel.Weights[0][n].Count(); m++)
                                for (int j = 0; j < kernel.Weights[0][n][m].GetLength(0); j++)
                                    for (int k = 0; k < kernel.Weights[0][n][m].GetLength(1); k++)
                                        kernel.Weights[0][n][m][j, k] = double.Parse(reader.ReadLine());
                        Kernels.Add(kernel);
                        break;
                    case LayerType.Hidden:
                        layer = new Layer(int.Parse(reader.ReadLine()), LayerType.Hidden, (ActivationFcn)Enum.Parse(typeof(ActivationFcn), reader.ReadLine()));
                        Layers.Add(layer);
                        if (i == 0)
                        {
                            if (CNNtype != CNNType.NN)
                                kernel = new Kernel(layer, Layers[Layers.Count - 2].HiddenCount, 1);
                            else
                                kernel = new Kernel(layer, pc, 1);
                        }
                        else
                            kernel = new Kernel(layer, Layers[Layers.Count - 2].HiddenCount, 1);
                        for (int j = 0; j < kernel.Bias.GetLength(0); j++)
                            kernel.Bias[j, 0] = double.Parse(reader.ReadLine());
                        for (int j = 0; j < kernel.Weights[0][0][0].GetLength(0); j++)
                            for (int k = 0; k < kernel.Weights[0][0][0].GetLength(1); k++)
                                kernel.Weights[0][0][0][j, k] = double.Parse(reader.ReadLine());
                        Kernels.Add(kernel);
                        break;
                    case LayerType.Pool:
                        layer = new Layer(int.Parse(reader.ReadLine()), int.Parse(reader.ReadLine()), (PoolType)Enum.Parse(typeof(PoolType), reader.ReadLine()), Layers[Layers.Count - 1].KernelCount);
                        Layers.Add(layer);
                        break;
                }
            }
            reader.Close();
        }

        /// <summary>
        /// сохранить нейросеть в файл
        /// </summary>
        /// <param name="path"></param>
        public void SaveCNN(string path)
        {
            StreamWriter writer = new StreamWriter(path);
            writer.WriteLine(Iterations);
            writer.WriteLine(CNNtype);
            if (CNNtype == CNNType.NN)
                writer.WriteLine(dataset.Data[0][0].GetLength(0));
            writer.WriteLine(Layers.Count);
            int NoKernelsCount = 0;
            int P = 1;
            for (int i = 0; i < Layers.Count; i++)
            {
                writer.WriteLine(Layers[i].Type);
                switch (Layers[i].Type)
                {
                    case LayerType.Convolutional:
                        writer.WriteLine(Layers[i].KernelCount);
                        writer.WriteLine(Layers[i].KernelWidth);
                        writer.WriteLine(Layers[i].KernelHeight);
                        writer.WriteLine(Layers[i].StepRight);
                        writer.WriteLine(Layers[i].StepDown);
                        writer.WriteLine(Layers[i].CombineChannels);
                        writer.WriteLine(Layers[i].ActFcn);
                        writer.WriteLine(Kernels[i - NoKernelsCount].Weights[0][0].Count());
                        for (int k = 0; k < Kernels[i - NoKernelsCount].Bias.GetLength(0); k++)
                            for (int n = 0; n < Kernels[i - NoKernelsCount].Bias.GetLength(1); n++)
                                writer.WriteLine(Kernels[i - NoKernelsCount].Bias[k, n]);
                        if (i != 0) P = Layers[i - 1].KernelCount;
                        for (int p = 0; p < P; p++)
                            for (int j = 0; j < Layers[i].KernelCount; j++)
                                for (int k = 0; k < Kernels[i - NoKernelsCount].Weights[p][j].Count(); k++)
                                    for (int n = 0; n < Layers[i].KernelWidth; n++)
                                        for (int m = 0; m < Layers[i].KernelHeight; m++)
                                            writer.WriteLine(Kernels[i - NoKernelsCount].Weights[p][j][k][n, m]);
                        break;
                    case LayerType.FullConnected:
                        writer.WriteLine(Layers[i].HiddenCount);
                        writer.WriteLine(Layers[i].ActFcn);
                        writer.WriteLine(Kernels[i - NoKernelsCount].Weights[0][0][0].GetLength(0));
                        writer.WriteLine(Kernels[i - NoKernelsCount].Weights[0][0][0].GetLength(1));
                        for (int j = 0; j < Kernels[i - NoKernelsCount].Bias.GetLength(0); j++)
                            writer.WriteLine(Kernels[i - NoKernelsCount].Bias[j, 0]);
                        for (int n = 0; n < Layers[i].HiddenCount; n++)
                            for (int m = 0; m < Kernels[i - NoKernelsCount].Weights[0][n].Count(); m++)
                                for (int j = 0; j < Kernels[i - NoKernelsCount].Weights[0][n][m].GetLength(0); j++)
                                    for (int k = 0; k < Kernels[i - NoKernelsCount].Weights[0][n][m].GetLength(1); k++)
                                        writer.WriteLine(Kernels[i - NoKernelsCount].Weights[0][n][m][j, k]);
                        break;
                    case LayerType.Hidden:
                        writer.WriteLine(Layers[i].HiddenCount);
                        writer.WriteLine(Layers[i].ActFcn);
                        for (int j = 0; j < Kernels[i - NoKernelsCount].Bias.GetLength(0); j++)
                            writer.WriteLine(Kernels[i - NoKernelsCount].Bias[j, 0]);
                        for (int j = 0; j < Kernels[i - NoKernelsCount].Weights[0][0][0].GetLength(0); j++)
                            for (int k = 0; k < Kernels[i - NoKernelsCount].Weights[0][0][0].GetLength(1); k++)
                                writer.WriteLine(Kernels[i - NoKernelsCount].Weights[0][0][0][j, k]);
                        break;
                    case LayerType.Pool:
                        writer.WriteLine(Layers[i].PoolWidth);
                        writer.WriteLine(Layers[i].PoolHeight);
                        writer.WriteLine(Layers[i].PoolT);
                        NoKernelsCount++;
                        break;
                }
            }
            writer.Flush();
            writer.Close();
        }

        double[][][,] Pool(Layer layer, double[][][,] input)//слой субдискретизации
        {
            int FMaps = input.Count();
            int Channels = input[0].Count();
            int MapWidth = input[0][0].GetLength(0);
            int MapHeight = input[0][0].GetLength(1);
            int NewWidth = MapWidth / layer.PoolWidth;
            int NewHeight = MapHeight / layer.PoolHeight;
            double[][][,] output = new double[FMaps][][,];
            Parallel.For(0, FMaps, i =>
            {
                output[i] = new double[Channels][,];
                for (int j = 0; j < Channels; j++)
                    output[i][j] = new double[NewWidth, NewHeight];

            });
            Parallel.For(0, FMaps, i =>//количество карт признаков
            {
                for (int j = 0; j < Channels; j++)//количество цветовых каналов
                    for (int k = 0; k < NewWidth; k++)//ширина выходных карт признаков
                        for (int m = 0; m < NewHeight; m++)//высота выходных карт признаков
                        {
                            if (layer.PoolT == PoolType.Avg)
                                output[i][j][k, m] = 0;
                            else
                                output[i][j][k, m] = double.MinValue;
                            for (int n = 0; n < layer.PoolWidth; n++)//ширина окна субдискретизации
                                for (int f = 0; f < layer.PoolHeight; f++)//высота окна субдискретизации
                                {
                                    if (layer.PoolT == PoolType.Avg)
                                        output[i][j][k, m] += input[i][j][k * layer.PoolWidth + n, m * layer.PoolHeight + f];
                                    else
                                    {
                                        if (output[i][j][k, m] < input[i][j][k * layer.PoolWidth + n, m * layer.PoolHeight + f])
                                            output[i][j][k, m] = input[i][j][k * layer.PoolWidth + n, m * layer.PoolHeight + f];
                                    }
                                }
                            if (layer.PoolT == PoolType.Avg)
                                output[i][j][k, m] /= (layer.PoolWidth * layer.PoolHeight);
                        }
            });
            return output;
        }

        double[][][,] Convolution(Layer layer, Kernel kernel, double[][][,] input)//сверточный слой
        {
            int FMaps = input.Count();
            int Channels = input[0].Count();
            int MapWidth = input[0][0].GetLength(0);
            int MapHeight = input[0][0].GetLength(1);
            int NewMaps = layer.KernelCount;
            int NewChannels = Channels;
            if (layer.CombineChannels)
                NewChannels = 1;
            int NewWidth = (MapWidth - layer.KernelWidth) / layer.StepRight + 1;
            int NewHeight = (MapHeight - layer.KernelHeight) / layer.StepDown + 1;
            double[][][,] output = new double[NewMaps][][,];
            Parallel.For(0, NewMaps, i =>
            {
                output[i] = new double[NewChannels][,];
                for (int j = 0; j < NewChannels; j++)
                {
                    output[i][j] = new double[NewWidth, NewHeight];
                    for (int k = 0; k < NewWidth; k++)
                        for (int m = 0; m < NewHeight; m++)
                            output[i][j][k, m] = 0;
                }
            });

            Parallel.For(0, NewMaps, i =>//количество выходных карт признаков
            {
                for (int j = 0; j < NewChannels; j++)//количество выходных цветовых каналов
                    for (int k = 0; k < NewWidth; k++)//ширина выходных карт признаков
                        for (int m = 0; m < NewHeight; m++)//высота выходных карт признаков
                        {
                            output[i][j][k, m] = kernel.Bias[i, j];
                            for (int a = 0; a < FMaps; a++)//количество входных карт признаков
                                for (int b = 0; b < Channels; b++)//количество входных цветовых каналов
                                    for (int c = 0; c < layer.KernelWidth; c++)//ширина ядер свертки
                                        for (int d = 0; d < layer.KernelHeight; d++)//высота ядер свертки
                                        {
                                            if (NewChannels == 1)
                                                output[i][j][k, m] += kernel.Weights[a][i][b][c, d] * input[a][b][k * layer.StepRight + c, m * layer.StepDown + d];
                                            if (NewChannels != 1 && b == 0)
                                                output[i][j][k, m] += kernel.Weights[a][i][j][c, d] * input[a][j][k * layer.StepRight + c, m * layer.StepDown + d];
                                        }
                            output[i][j][k, m] = Activation(output[i][j][k, m], layer.ActFcn);
                        }
            });
            return output;
        }

        double[][][,] FullConnect(Layer layer, Kernel kernel, double[][][,] input)//слой перехода от карт признаков к одномерным слоям
        {
            int FMaps = input.Count();
            int MapWidth = input[0][0].GetLength(0);
            int MapHeight = input[0][0].GetLength(1);
            int HiddenCnt = layer.HiddenCount;
            double[][][,] output = new double[1][][,];
            output[0] = new double[1][,];
            output[0][0] = new double[HiddenCnt, 1];
            Parallel.For(0, HiddenCnt, i =>//количество выходных нейронов
            {
                output[0][0][i, 0] = kernel.Bias[i, 0];
                for (int j = 0; j < FMaps; j++)//количество входных карт признаков
                    for (int k = 0; k < MapWidth; k++)//ширина входных карт признаков
                        for (int m = 0; m < MapHeight; m++)//высота входных карт признаков
                            output[0][0][i, 0] += input[j][0][k, m] * kernel.Weights[0][i][j][k, m];
                output[0][0][i, 0] = Activation(output[0][0][i, 0], layer.ActFcn);

            });
            return output;
        }

        double[][][,] Hidden(Layer layer, Kernel kernel, double[][][,] input)//скрытый слой
        {
            int InputNeurons = input[0][0].GetLength(0);
            int OutputNeurons = layer.HiddenCount;
            double[][][,] output = new double[1][][,];
            output[0] = new double[1][,];
            output[0][0] = new double[OutputNeurons, 1];
            Parallel.For(0, OutputNeurons, i =>//количество выходных нейронов
            {
                output[0][0][i, 0] = kernel.Bias[i, 0];
                for (int j = 0; j < InputNeurons; j++)//количество входных нейронов
                    output[0][0][i, 0] += input[0][0][j, 0] * kernel.Weights[0][0][0][i, j];
                output[0][0][i, 0] = Activation(output[0][0][i, 0], layer.ActFcn);

            });
            return output;
        }

        double[] OutputToArray(double[][][,] Output)//преобразование выходных данных к одномерному массиву
        {
            double[] Result = new double[Output[0][0].GetLength(0)];
            int Count = Result.Count();
            Parallel.For(0, Count, i =>
            {
                Result[i] = Output[0][0][i, 0];
            });
            return Result;
        }

        double Activation(double x, ActivationFcn act)//функция активации
        {
            switch (act)
            {
                case ActivationFcn.ReLU:
                    if (x < 0)
                        return x * 0.01;
                    else
                        return x;
                case ActivationFcn.Sigm:
                    return 1 / (1.0 + Math.Exp(-x));
                case ActivationFcn.Tanh:
                    return Math.Tanh(x);
                case ActivationFcn.TanhConv:
                    return 1.7159 * Math.Tanh(x * 2.0 / 3.0);
                case ActivationFcn.TanhLinear://экспериментальная функция
                    if (x < 2 && x > -2)
                        return Math.Tanh(x);
                    if (x >= 2)
                        return (0.945 + x * 0.01);
                    if (x <= -2)
                        return (-0.945 + x * 0.01);
                    break;
            }
            return 0;
        }

        double derivActivation(double x, ActivationFcn act)//производная функции активации (определяет значение производной по значению функции активации)
        {//работает только с монотонными функциями
            switch (act)
            {
                case ActivationFcn.ReLU:
                    if (x < 0)
                        return 0.01;
                    else
                        return 1;
                case ActivationFcn.Sigm:
                    return (1 - (x)) * (x);
                case ActivationFcn.Tanh:
                    return 1 - x * x;
                case ActivationFcn.TanhConv:
                    return 1.14393333 - 0.38852302857 * x * x;
                case ActivationFcn.TanhLinear:
                    if (x < 0.965 && x > -0.965)
                        return 1 - x * x;
                    else
                        return 0.01;
            }
            return 0;
        }
    }
}
