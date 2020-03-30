// neural network post-processing

//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//using UnityEngine.Rendering;

//namespace NNPP
//{
//    public class UpSampling2D : NNLayerBase
//    {
//        public Vector2Int Size;
//        private ComputeBuffer outputbuffer;
//        public int AlternativeInputId;
//        public int kernal_last_layer, kernal_down_layer;
//        public Vector3Int DownShape;

//        public UpSampling2D() : base()
//        {
//            KernelId = NNCompute.Instance.Kernel("UpSampling2D");
//        }
//        public override void Init(Vector3Int inputShape)
//        {
//            InputShape = inputShape;

//            if (AlternativeInputId == 0)
//            {
//                OutputShape = new Vector3Int(inputShape.x * Size.x, inputShape.y * Size.y, inputShape.z);
//            }
//            else
//            {
//                OutputShape = new Vector3Int(DownShape.x, DownShape.y, kernal_down_layer + kernal_last_layer);
//            }
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

//            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput1", input[0] as ComputeBuffer);
//            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[1] as ComputeBuffer);  // cat(Down, last_layer)
//            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);

//            //NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
//            //NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);

//            NNCompute.Instance.Shader.SetInts("InputShape", new int[3]
//            {
//                InputShape.x,
//                InputShape.y,
//                kernal_last_layer
//            });
//            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
//            {
//                InputShape.y * kernal_last_layer,
//                kernal_last_layer,
//                1
//            });

//            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier1", new int[3]
//           {
//                InputShape.y * kernal_down_layer,
//                kernal_down_layer,
//                1
//           });

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
//            NNCompute.Instance.Shader.SetInts("Size", new int[2]
//            {
//                Size.x,
//                Size.y
//            });
//            NNCompute.Instance.Shader.Dispatch(KernelId, Mathf.CeilToInt(OutputShape.x / 8.0f), Mathf.CeilToInt(OutputShape.y / 8.0f), OutputShape.z);
//        }
//    }
//}

//neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class UpSampling2D : NNLayerBase
    {
        public Vector2Int Size;
        private ComputeBuffer outputbuffer;
        public int AlternativeInputId;
        public int mode;
        public Vector3Int DownShape;
        public int kernal_last_layer, kernal_down_layer;

        public UpSampling2D() : base()
        {
            KernelId = NNCompute.Instance.Kernel("UpSampling2D");
        }
        public override void Init(Vector3Int inputShape)
        {
            InputShape = inputShape;

            if (AlternativeInputId == 0)
            {
                OutputShape = new Vector3Int(inputShape.x * Size.x, inputShape.y * Size.y, inputShape.z);
            }
            else
            {
                OutputShape = new Vector3Int(DownShape.x, DownShape.y, DownShape.z);
            }
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
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
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
            NNCompute.Instance.Shader.SetInts("Size", new int[2]
            {
                Size.x,
                Size.y
            });
            NNCompute.Instance.Shader.Dispatch(KernelId, Mathf.CeilToInt(OutputShape.x / 16.0f), Mathf.CeilToInt(OutputShape.y / 16.0f), OutputShape.z);
        }
    }
}