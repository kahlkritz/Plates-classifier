using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Collections;

namespace WRCI_Assignment2
{
    class Program
    {
        static void Main(string[] args)
        {
            double valSSE = 0;
            double valMSE = 0;
            int valCounter = 0;
            int epochs = 100;
            int testCounter = 0;
            double testSSE = 0;
            double testMSE = 0;
            int epochCounter = 0;
            int patternCount = 1900;
            double SSE = 100000000; //Sum Square Error
            double MSE = 0; // Mean Squared Error
            double learningRate = 0.1; 
            
            int hiddenLayerSize = 17;    //Allows for changing the architecture as needed
            double initialWeightSizeFactor = 0.1;   //Allows for changing the weight inital values appropriately
            double[,] patterns = new double[patternCount, 28];   //Patterns (plates)
            double[,] V = new double[hiddenLayerSize, 28];  //Input to hidden weights
            double[] Yj = new double[hiddenLayerSize];  //Hidden layer neuron outputs
            double[,] W = new double[7, hiddenLayerSize+1];   //Hidden to output weights
            double[] Ok = new double[7];    //Actual Outputs
            int[] Tk = new int[7];    //Expected outputs
            double[] outErrors = new double[7]; //Holds the amount of prediction errors
            double accCount = 0;    //Counter that holds the amount of correct predictions
            double testAcc = 0;

//Read from .csv file to patterns[]
            StreamReader SR = new StreamReader("PlatesScaled.csv");
            for (int row = 0; row < patternCount; row++)    //Loops through rows of the training set: Patters P
            {
                string[] data = (SR.ReadLine()).Split(',');
                for (int col = 0; col < 28; col++)  //Loops through collumns of the training set: Inputs Zi
                {
                    patterns[row, col] = double.Parse(data[col]);   //Populate patterns[]
                }  
            }
//Populate initial weight arrays with appropriate random values
            Random random = new Random();
            //Populate V
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                for (int i = 0; i < 28; i++)
                {
                    V[j, i] = random.NextDouble() * initialWeightSizeFactor;
                }
            }
            //Populate W
            for (int k = 0; k < 7; k++)
            {
                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    W[k, j] = random.NextDouble() * initialWeightSizeFactor;
                }
            }

//GRADIENT DECENT ALGORITHM
            StreamWriter SW = new StreamWriter("SSE.txt");
            while (epochs > 0)
            {
                SSE = 0;    //Reinitialise MSError to 0 after each iteration
                accCount = 0;   //Reinitilise the correct amount of predictions
                for (int y = 0; y < 7; y++)
                    outErrors[y] = 0;   //Reinitialise error array to o's
                for (int p = 0; p < patternCount; p++)   //Loop through the patterns
                {
                    Tk = convert(Convert.ToInt32(patterns[p, 27])); //Convert the target integer to a n array of "binaries"
                    //UPDATE Yj
                    for (int j = 0; j < hiddenLayerSize; j++)   //Loop through hidden layer
                    {
                        double netY = 0;    //Clear net after each hidden layer neuron has fired
                        for (int i = 0; i < 27; i++)    //Loop through inputs
                        {
                            netY += V[j, i] * patterns[p,i];    //Add each input multiplied with appropriate weight
                        }
                        netY += -1 * V[j, 27];  //Add bias input
                        Yj[j] = sigmoid(netY);  //Fire neuron after the appropriate sum is found
                    }
                    //UPDATE Ok
                    for (int k = 0; k < 7; k++)
                    {
                        double netO = 0;
                        for (int j = 0; j < hiddenLayerSize; j++)
                        {
                            netO += Yj[j] * W[k, j];    
                        }
                        netO += -1 * W[k, hiddenLayerSize];
                        Ok[k] = sigmoid(netO);
                    }
                    //Update W
                    for(int k = 0; k < 7; k++)
                    {
                        for(int j = 0; j < hiddenLayerSize; j++)
                        {
                            W[k, j] = W[k,j] - (learningRate * delta_W(Tk[k], Ok[k], Yj[j]));
                        }
                    }
                    //Update V
                    for (int i = 0; i < 27; i++)
                    {
                        for (int j = 0; j < hiddenLayerSize; j++)
                        {
                            for (int k = 0; k < 7; k++)
                            {
                                V[j, i] = V[j,i] - (learningRate * delta_V(Tk[k], Ok[k], Yj[j], patterns[p, i], W[k, j]));
                            }
                        }
                    }
//VALIDATION SET    
                    if ((p + 1) % 5 == 0)
                    {
                        int pV = valCounter + 1500;
                        Tk = convert(Convert.ToInt32(patterns[pV, 27])); //Convert the target integer to a n array of "binaries"
                        //UPDATE Yj
                        for (int j = 0; j < hiddenLayerSize; j++)   //Loop through hidden layer
                        {
                            double netY = 0;    //Clear net after each hidden layer neuron has fired
                            for (int i = 0; i < 27; i++)    //Loop through inputs
                            {
                                netY += V[j, i] * patterns[pV, i];    //Add each input multiplied with appropriate weight
                            }
                            netY += -1 * V[j, 27];  //Add bias input
                            Yj[j] = sigmoid(netY);  //Fire neuron after the appropriate sum is found
                        }
                        //UPDATE Ok
                        for (int k = 0; k < 7; k++)
                        {
                            double valNetO = 0;
                            for (int j = 0; j < hiddenLayerSize; j++)
                            {
                                valNetO += Yj[j] * W[k, j];
                            }
                            valNetO += -1 * W[k, hiddenLayerSize];
                            Ok[k] = sigmoid(valNetO);
                        }
                        //Calculate Error
                        for (int k = 0; k < 7; k++)
                        {
                            valSSE += Math.Pow(Tk[k] - Ok[k], 2);
                        }
                        valCounter++;
                    }
                    valMSE = valSSE / (300 * 7);
                    //Calculate Error
                    for (int k = 0; k < 7; k++)
                    {
                        SSE += Math.Pow(Tk[k] - Ok[k], 2);
                    }
                    //Count accurate perdictions
                    double maxOut = Ok.Max();   //Find the maximum value in output array (actual output array)
                    if (patterns[p,27] == (Ok.ToList().IndexOf(maxOut) + 1))    //If the target is equal to the maximum in output array (actual output array)
                    {
                        int maxIndex = Ok.ToList().IndexOf(maxOut); //Find the index of the maximum calculated value
                        outErrors[maxIndex]++;  //Increment the "error" index
                        accCount++; //increment the correct prediction counter
                    } 
                }
                SW.WriteLine(SSE);
                MSE = SSE / (2900 * 7); //Divide total error by the 1500*7 to get the mean
                epochs--; //decrement iteration counter
                epochCounter++;
                
            }
            SW.Close();
//TEST SET
            for(int pV = 1800; pV < 1900; pV++)
            {
                int[] TkVal = convert(Convert.ToInt32(patterns[pV, 27])); //Convert the target integer to a n array of "binaries"
                //UPDATE Yj
                for (int j = 0; j < hiddenLayerSize; j++)   //Loop through hidden layer
                {
                    double netY = 0;    //Clear net after each hidden layer neuron has fired
                    for (int i = 0; i < 27; i++)    //Loop through inputs
                    {
                        netY += V[j, i] * patterns[pV, i];    //Add each input multiplied with appropriate weight
                    }
                    netY += -1 * V[j, 27];  //Add bias input
                    Yj[j] = sigmoid(netY);  //Fire neuron after the appropriate sum is found
                }
                //UPDATE Ok
                for (int k = 0; k < 7; k++)
                {
                    double valNetO = 0;
                    for (int j = 0; j < hiddenLayerSize; j++)
                    {
                        valNetO += Yj[j] * W[k, j];
                    }
                    valNetO += -1 * W[k, hiddenLayerSize];
                    Ok[k] = sigmoid(valNetO);
                }
                //Calculate Error
                for (int k = 0; k < 7; k++)
                {
                    testSSE += Math.Pow(TkVal[k] - Ok[k], 2);
                }
                testCounter++;
                double maxOut = Ok.Max();   //Find the maximum value in output array (actual output array)
                if (patterns[pV, 27] == (Ok.ToList().IndexOf(maxOut) + 1))    //If the target is equal to the maximum in output array (actual output array)
                {
                    int maxIndex = Ok.ToList().IndexOf(maxOut); //Find the index of the maximum calculated value
                    outErrors[maxIndex]++;  //Increment the "error" index
                    testAcc++; //increment the correct prediction counter
                }
            }
            //FINAL//////////////////////////////////////////////////
            StreamWriter sw1 = new StreamWriter("Final.txt");
            for (int pV = 1900; pV < 1940; pV++)
            {
                StreamReader SR1 = new StreamReader("PlatesScaled.csv");
                for (int row = 0; row < 1940; row++)    //Loops through rows of the training set: Patters P
                {
                    string[] data = (SR1.ReadLine()).Split(',');
                    for (int col = 0; col < 27; col++)  //Loops through collumns of the training set: Inputs Zi
                    {
                        patterns[row, col] = double.Parse(data[col]);   //Populate patterns[]
                    }
                }
                //UPDATE Yj
                for (int j = 0; j < hiddenLayerSize; j++)   //Loop through hidden layer
                {
                    double netY = 0;    //Clear net after each hidden layer neuron has fired
                    for (int i = 0; i < 27; i++)    //Loop through inputs
                    {
                        netY += V[j, i] * patterns[pV, i];    //Add each input multiplied with appropriate weight
                    }
                    netY += -1 * V[j, 27];  //Add bias input
                    Yj[j] = sigmoid(netY);  //Fire neuron after the appropriate sum is found
                }
                //UPDATE Ok
                for (int k = 0; k < 7; k++)
                {
                    double valNetO = 0;
                    for (int j = 0; j < hiddenLayerSize; j++)
                    {
                        valNetO += Yj[j] * W[k, j];
                    }
                    valNetO += -1 * W[k, hiddenLayerSize];
                    Ok[k] = sigmoid(valNetO);
                }

                double maxOut = Ok.Max();   //Find the maximum value in output array (actual output array)
               
                int maxIndex = Ok.ToList().IndexOf(maxOut); //Find the index of the maximum calculated value
                Console.WriteLine("Pattern{0}: {1}", pV - 1899, maxIndex+1);
                sw1.WriteLine("Pattern{0}: {1}", pV-1899, maxIndex+1);
            }
            Console.ReadLine();
            testMSE = testSSE / (100 * 7);
            DisplayResults(SSE, MSE, accCount, outErrors, epochCounter, testSSE, testMSE, testAcc);
        }



        ////////////METHODS/////////////////////////////METHODS///////////////////////////METHODS//////////////////////////////////////////////////

        //Display final results
        
        public static void DisplayResults(double SSE, double MSError, double accCount, double[] outErrors, int iterations, double testSSE, double testMSE, double testAcc)
        {
            //Console.WriteLine("Test Set:");
            //Console.WriteLine("Test MSE: {0:f4}", testMSE);
            Console.WriteLine("Test SSE: {0:f4}", testSSE);
            Console.WriteLine("Training SSEror: {0:f6}", SSE);
            
            Console.WriteLine();
            Console.WriteLine();
            //Console.WriteLine("Traing Set");
            Console.WriteLine("MSError: {0:f4}", MSError);
            Console.WriteLine("Test Accurcy: {0:f3}", testAcc);
            Console.WriteLine("Accuracy: {0:f3}%", (accCount / 2900) * 100);
            Console.WriteLine("Iterations: {0}", iterations);
            Console.WriteLine();

            for (int x = 0; x < outErrors.Length; x++)
            {
                Console.WriteLine("Category {0} faults: {1} ", x + 1, outErrors[x]);
            }
            Console.ReadLine();
        }

        private static void disp2DArray(int listSize, double[,] array)
        {
            if (array.GetLength(0) == 2)
                listSize = array.GetLength(0);
            for (int rowdisp = 0; rowdisp < listSize; rowdisp++)
            {
                for (int colDisp = 0; colDisp < array.GetLength(1); colDisp++)
                {
                    Console.Write(array[rowdisp, colDisp]);
                    Console.Write(",");
                }
                Console.WriteLine();
            }
            Console.ReadLine();
        }

        //SIGMOID ACTIVATION FUNCTION
        private static double sigmoid(double net)
        {
            return 1 / (1 + Math.Exp(-net));
        }

        //CONVERT INT TO APPROPRIATE CATEGORY OUTPUT
        private static int[] convert(int categoryInt)
        {
            int[] catArray = new int[7];
            for (int x = 0; x < 7; x++)
            {
                if (categoryInt == (x+1))
                    catArray[x] = 1;
                else
                    catArray[x] = 0;
            }
            return catArray;
        }
        //Delta W
        private static double delta_W(double Tk, double Ok, double Yj)
        {
            return -1 * (Tk - Ok) * Ok * (1 - Ok) * Yj;
        }
        //Delta V
        private static double delta_V(double Tk, double Ok, double Yj, double Zi, double Wkj)
        {
            
            return delta_W(Tk,Ok,Yj) * Wkj * (1 - Yj) * Zi;
        }
        //Get maximum calculated value
        private static double getMax(double[] Ok)
        {
            double currentMax = -1000000;
            int maxIndex = 0;
            for(int k = 0; k < Ok.Length; k++)
            {
                if (Ok[k] > currentMax)
                {
                    currentMax = Ok[k];
                    maxIndex = k;
                }
            }
            return currentMax;
        }
        //Get index of max value
        private static int getMaxIndex(double[] Ok)
        {
            int maxIndex = 0;
            double maxVal = -100000000000000;
            for(int k = 0; k < Ok.Length; k++)
            {
                if (Ok[k] > maxVal)
                    maxIndex = k;
            }
            return maxIndex;
        }
        
    }
}
