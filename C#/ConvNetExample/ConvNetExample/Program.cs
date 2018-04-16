using System;
using System.Collections.Generic;
using System.Linq;
using ConvNet;

namespace ConvNetExample
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = "D:/2/Train/";
            Dataset ds_test = new Dataset(10, ColorModel.Grayscale, 32, 32);
            Dataset ds_train=new Dataset(10, ColorModel.Grayscale, 32, 32);
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 120; j++)//each of 10 directories contains 120 images
                    if (j % 6 != 0)
                        ds_train.AddImage(path + i + "/" + j + ".jpg", i, false, false, false);
                    else
                        ds_test.AddImage(path + i + "/" + j + ".jpg", i, false, false, false);
            ds_test.PrepareData();
            ds_train.PrepareData();
            CNN cnn = new CNN(CNNType.CNN);
            cnn.LoadData(ds_train);
            cnn.AddLayer(new Layer(15, 5, 5, 1, 1, false, ActivationFcn.ReLU));//15 kernels 5x5
            cnn.AddLayer(new Layer(2, 2, PoolType.Max, 15));//max pooling 2x2
            cnn.AddLayer(new Layer(20, 5, 5, 1, 1, false, ActivationFcn.ReLU));//20 kernels 5x5
            cnn.AddLayer(new Layer(2, 2, PoolType.Max, 20));//max pooling 2x2
            cnn.AddLayer(new Layer(50, LayerType.FullConnected, ActivationFcn.ReLU));
            cnn.AddLayer(new Layer(10, LayerType.Hidden, ActivationFcn.Tanh));
            cnn.CreateNewCNN();
            int Epochs = 10;
            int TrainCount = ds_train.Data.Count();
            Queue<Out> res = new Queue<Out>();
            Out Res;
            double cost = 0;
            int T = 0;
            int F = 0;
            cnn.Epohs = 0;
            cnn.Iterations = 0;
            cnn.LearnRate = 0.01;
            for(int i=0;i<Epochs;i++)//training
            {
                cnn.MixDataset();
                for (int j = 0; j < TrainCount; j++)
                {
                    Res = cnn.BackPropagate(j);
                    res.Enqueue(Res);
                    cost += Res.Cost;
                    if (Res.RecognRes)
                        T++;
                    else
                        F++;
                    cnn.Iterations++;
                    if(cnn.Epohs>0)
                    {
                        Res = res.Dequeue();
                        cost -= Res.Cost;
                        if (Res.RecognRes)
                            T--;
                        else
                            F--;
                    }
                    if (cnn.Iterations % 100 == 0)
                    {
                        if (cnn.Epohs > 0)
                        {                            
                            Console.WriteLine("Epoch: " + cnn.Epohs + " Iteration: " + cnn.Iterations);
                            Console.WriteLine("Loss: " + String.Format("{0:f4}", cost / TrainCount) + " Accuracy: " + String.Format("{0:f2}", 100.0 * T / (T + F)) + "%");
                            Console.WriteLine();
                        }
                        else
                        {
                            Console.WriteLine("Epoch: " + cnn.Epohs + " Iteration: " + cnn.Iterations);
                            Console.WriteLine("Loss: " + String.Format("{0:f4}", cost / cnn.Iterations) + " Accuracy: " + String.Format("{0:f2}", 100.0 * T / (T + F)) + "%");
                            Console.WriteLine();
                        }
                    }
                }
                cnn.Epohs++;
                cnn.LearnRate *= cnn.LearnRateDecrease;
            }

            T = 0;
            F = 0;
            int TestCount = ds_test.Data.Count();
            double[] result;
            for(int i=0;i<TestCount;i++)//testing
            {
                result = cnn.Run(ds_test.Data[i]);
                if (result.Max() == result[ds_test.Answers[i]])
                    T++;
                else
                    F++;
                Console.WriteLine(" Accuracy: " + String.Format("{0:f2}", 100.0 * T / (T + F)) + "%");
            }
            Console.ReadKey();

        }
    }
}
