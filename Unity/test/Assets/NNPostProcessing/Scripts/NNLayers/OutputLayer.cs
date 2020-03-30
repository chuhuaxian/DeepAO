// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;
namespace NNPP
{
    public class OutputLayer : NNLayerBase
    {
        public RenderTexture outputTex;
        public OutputLayer() : base()
        {
            KernelId = NNCompute.Instance.Kernel("OutputLayer");
        }

        public override void Run(object[] input)
        {
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
            NNCompute.Instance.Shader.SetTexture(KernelId, "OutputImage", outputTex);
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
            NNCompute.Instance.Shader.Dispatch(KernelId, OutputShape.x / 8, OutputShape.y / 8, 1);


            //RenderTexture.active = outputTex;
            //Texture2D t2d;
            //int num = 0;
            //t2d = new Texture2D(outputTex.width, outputTex.height, TextureFormat.RGBAFloat, false);
            //t2d.ReadPixels(new Rect(0, 0, outputTex.width, outputTex.height), 0, 0);
            //t2d.Apply();

            //RenderTexture.active = null;
            //byte[] bytes = t2d.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);

            ////string u = string.Format("1{0:D4}", num);

            //File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-out.exr", bytes);


            //Debug.Log("当前截图序号为++++++++++++：" + num.ToString());

        }

        public override void Init(Vector3Int inputShape)
        {
            base.Init(inputShape);
            if (outputTex != null)
                outputTex.Release();
            outputTex = new RenderTexture(OutputShape.y, OutputShape.x, 32, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            //outputTex = new RenderTexture(OutputShape.y, OutputShape.x, 32, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            outputTex.enableRandomWrite = true;
            outputTex.Create();
        }

        public override void Release()
        {
            if (outputTex != null)
                outputTex.Release();
        }
    }
}