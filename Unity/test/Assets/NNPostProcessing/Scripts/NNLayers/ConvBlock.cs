// neural network post-processing

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    [System.Serializable]
    public class ConvBlock : NNLayerBase
    {
        public float[] weightcache;
        public int Filters;
        public Vector2Int KernalSize;
        public Vector2Int Stride;
        private ComputeBuffer outputbuffer;

        private ComputeBuffer weightbuffer;
        public ConvBlock() : base()
        {
            KernelId = NNCompute.Instance.KernelConv2D(32);
        }

        public override void FromCache()
        {
            weightbuffer = new ComputeBuffer(weightcache.Length, sizeof(float));
            weightbuffer.SetData(weightcache);
            weightcache = null;
        }

        public override void Init(Vector3Int inputShape)
        {
            InputShape = inputShape;
            OutputShape = new Vector3Int(inputShape.x / Stride.x, inputShape.y / Stride.y, Filters);
            if (outputbuffer != null)
                outputbuffer.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            int maxfilter = Mathf.Max(inputShape.z, Filters);
            KernelId = NNCompute.Instance.KernelConv2D(maxfilter);
            Output = outputbuffer;
        }

        public override void Release()
        {
            if (weightbuffer != null)
                weightbuffer.Release();
            if (outputbuffer != null)
                outputbuffer.Release();
        }

        public override void Run(object[] input)
        {
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer); //写入GPU， KernelId-> shader
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);

            NNCompute.Instance.Shader.SetBuffer(KernelId, "Weights", weightbuffer);
            NNCompute.Instance.Shader.SetInts("InputShape", new int[3]
            {
                InputShape.x,
                InputShape.y,
                InputShape.z
            });
            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * InputShape.z,
                InputShape.z,
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
            NNCompute.Instance.Shader.SetInts("WeightsShape", new int[4]
            {
                KernalSize.x,
                KernalSize.y,
                InputShape.z,
                Filters
            });
            NNCompute.Instance.Shader.SetInts("WeightsShapeIdMultiplier", new int[4]
            {
                KernalSize.y * InputShape.z * Filters,
                InputShape.z * Filters,
                Filters,
                1
            });
            NNCompute.Instance.Shader.SetInts("Stride", new int[2]
            {
                Stride.x,
                Stride.y
            });

            NNCompute.Instance.Shader.Dispatch(KernelId, 1, OutputShape.y, Mathf.CeilToInt(OutputShape.x / 4.0f)); // 计算并写回CPU


        }
    }
}