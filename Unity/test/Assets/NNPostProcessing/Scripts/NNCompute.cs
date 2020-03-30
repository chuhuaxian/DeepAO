// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NNPP
{
    public class NNCompute
    {
        private static NNCompute _instance;

        public static NNCompute Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new NNCompute();
                    _instance.Init();
                }

                return _instance;
            }

        }

        const int kernal_num = 5;

        private int[] Conv2DKernelLayers = new int[kernal_num] { 4, 8, 16, 32, 64};

        private int[] Conv2DKernels = new int[kernal_num];

        public ComputeShader Shader;
        private string shaderpath = "NNLayer";
        private int LeakyReluKernel, BatchNormalizationKernel, InputLayerKernel, AddKernel,
            ConcatenateKernel, OutputLayerKernel, UpSampling2DKernel, Maxpooling2DKernel, ReluKernel, TanhKernel, DebugLayerKernel, AvgPooling2DKernel;

        private void Init()
        {
            Conv2DKernels = new int[kernal_num];
            Shader = Resources.Load<ComputeShader>(shaderpath);
            for (int i = 0; i < Conv2DKernelLayers.Length; i++)
            {
                Conv2DKernels[i] = Shader.FindKernel(string.Format("Conv2D_{0}", Conv2DKernelLayers[i]));
            }
            LeakyReluKernel = Shader.FindKernel("LeakyReLU");
            BatchNormalizationKernel = Shader.FindKernel("BatchNormalization");
            InputLayerKernel = Shader.FindKernel("InputLayer");
            OutputLayerKernel = Shader.FindKernel("OutputLayer");
            DebugLayerKernel = Shader.FindKernel("DebugLayer");
            UpSampling2DKernel = Shader.FindKernel("UpSampling2D");
            Maxpooling2DKernel = Shader.FindKernel("MaxPooling2D");
            ConcatenateKernel = Shader.FindKernel("Concatenate");
            ReluKernel = Shader.FindKernel("ReLU");
            TanhKernel = Shader.FindKernel("Tanh");
            AddKernel = Shader.FindKernel("Add");
            AvgPooling2DKernel = Shader.FindKernel("AvgPooling2D");

        }

        public int KernelConv2D(int channel)
        {
            for (int i = 0; i < Conv2DKernelLayers.Length; i++)
            {
                if (channel <= Conv2DKernelLayers[i])
                {
                    return Conv2DKernels[i];
                }
            }
            return -1;
        }

        public int Kernel(string name)
        {
            switch (name)
            {
                case ("LeakyReLU"):
                    return LeakyReluKernel;
                case ("BatchNormalization"):
                    return BatchNormalizationKernel;
                case ("InputLayer"):
                    return InputLayerKernel;
                case ("OutputLayer"):
                    return OutputLayerKernel;
                case ("DebugLayer"):
                    return OutputLayerKernel;
                case ("UpSampling2D"):
                    return UpSampling2DKernel;
                case ("MaxPooling2D"):
                    return Maxpooling2DKernel;
                case ("AvgPooling2D"):
                    return Maxpooling2DKernel;
                case ("ReLU"):
                    return ReluKernel;
                case ("Tanh"):
                    return TanhKernel;
                case ("Concatenate"):
                    return ConcatenateKernel;
                case ("Add"):
                    return AddKernel;
                default:
                    return -1;
            }
        }
    }
}