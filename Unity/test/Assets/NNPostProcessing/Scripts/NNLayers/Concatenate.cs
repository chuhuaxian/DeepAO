﻿// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class Concatenate : NNLayerBase
    {
        private ComputeBuffer outputbuffer;
        public int AlternativeInputId;
        public int kernal_last_layer, kernal_down_layer;


        public Concatenate() : base()
        {
            KernelId = NNCompute.Instance.Kernel("Concatenate");
        }

        public override void Init(Vector3Int inputShape)
        {
            base.Init(inputShape);
            if (outputbuffer != null)
                outputbuffer.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            Output = outputbuffer;
        }

        public override void Release()
        {
            if (outputbuffer != null)
                outputbuffer.Release();
        }

        public override void Run(object[] input)
        {

            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput1", input[1] as ComputeBuffer);  // cat(Down, last_layer)
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
            NNCompute.Instance.Shader.SetInts("InputShape", new int[3]
            {
                InputShape.x,
                InputShape.y,
                kernal_down_layer
            });
            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * kernal_down_layer,
                kernal_down_layer,
                1
            });
            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier1", new int[3]
            {
                InputShape.y * kernal_last_layer,
                kernal_last_layer,
                1
            });
            NNCompute.Instance.Shader.SetInts("OutputShape", new int[3]
            {
                OutputShape.x,
                OutputShape.y,
                OutputShape.z
            });
            NNCompute.Instance.Shader.SetInts("OutputShapeIdMultiplier", new int[3]
            {
                OutputShape.y * OutputShape.z,
                OutputShape.z,
                1
            });
            NNCompute.Instance.Shader.Dispatch(KernelId, OutputShape.x / 16, OutputShape.y / 16, OutputShape.z);
        }
    }
}



//// neural network post-processing

//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//using UnityEngine.Rendering;

//namespace NNPP
//{
//    public class Concatenate : NNLayerBase
//    {
//        private ComputeBuffer outputbuffer;
//        public int AlternativeInputId;
//        public Concatenate() : base()
//        {
//            KernelId = NNCompute.Instance.Kernel("Concatenate");
//        }

//        public override void Init(Vector3Int inputShape)
//        {
//            base.Init(inputShape);
//            if (outputbuffer != null)
//                outputbuffer.Release();
//            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
//            Output = outputbuffer;
//        }

//        public override void Release()
//        {
//            if (outputbuffer != null)
//                outputbuffer.Release();
//        }

//        public override void Run(object[] input)
//        {
//            var input0 = input[0] as ComputeBuffer;
//            int inputfilters0 = input0.count / (OutputShape.x * OutputShape.y);
//            int inputfilters1 = OutputShape.z - inputfilters0;
//            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input0);
//            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput1", input[1] as ComputeBuffer);  // cat(Down, last_layer)
//            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
//            NNCompute.Instance.Shader.SetInts("InputShape", new int[3]
//            {
//                InputShape.x,
//                InputShape.y,
//                inputfilters0
//            });
//            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
//            {
//                InputShape.y * inputfilters0,
//                inputfilters0,
//                1
//            });
//            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier1", new int[3]
//            {
//                InputShape.y * inputfilters1,
//                inputfilters1,
//                1
//            });
//            NNCompute.Instance.Shader.SetInts("OutputShape", new int[3]
//            {
//                OutputShape.x,
//                OutputShape.y,
//                OutputShape.z
//            });
//            NNCompute.Instance.Shader.SetInts("OutputShapeIdMultiplier", new int[3]
//            {
//                OutputShape.y * OutputShape.z,
//                OutputShape.z,
//                1
//            });
//            NNCompute.Instance.Shader.Dispatch(KernelId, OutputShape.x / 8, OutputShape.y / 8, OutputShape.z);
//        }
//    }
//}