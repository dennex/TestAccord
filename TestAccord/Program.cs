using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Controls;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;

namespace TestAccord
{
    class Program
    {
        [MTAThread]
        static void Main(string[] args)
        {
            double[][] inputs ={
                                    new double[]{0,0},
                                    new double[]{1,0},
                                    new double []{0,1},
                                    new double[]{1,1},
                               };

            int[] outputs = {
                                -1,
                                +1,
                                +1,
                                -1,
                            };

            double[][] inputs2 ={
                                    new double[]{0,0},
                                    new double[]{0,1},
                                    new double []{1,0},
                                    new double[]{1,1},
                               };

            var ksvm = new KernelSupportVectorMachine(new Gaussian(), 2);
            var smo = new SequentialMinimalOptimization(machine: ksvm, inputs: inputs, outputs: outputs)
            {
                Complexity = 100
            };

            double error = smo.Run();

            Console.WriteLine("error:" + error);

            // Show results on screen 
            
            ScatterplotBox.Show("Training data", inputs, outputs);

            ScatterplotBox.Show("SVM results", inputs2,
                inputs2.Apply(p => System.Math.Sign(ksvm.Compute(p))));

            Console.ReadKey(); 
        }
    }
}
