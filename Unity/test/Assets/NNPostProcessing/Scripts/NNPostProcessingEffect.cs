// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;


namespace NNPP
{

    [System.Serializable]
    //[ExecuteInEditMode] //ExecuteInEditMode属性的作用是在EditMode下也可以执行脚本。Unity中默认情况下，脚本只有在运行的时候才被执行，加上此属性后，不运行程序，也能执行脚本。
    [RequireComponent(typeof(Camera))]
    public class NNPostProcessingEffect : MonoBehaviour
    {
        //public NNStyle style = NNStyle.starry_night;
        public NNStyle style = NNStyle.model_4;

        private NNModel model;
        private Material mMat;
        RenderTexture rt;
        //RenderTexture rt;
        Texture2D t2d;
        int num = 0;

        void Start()
        {
            //rt = new RenderTexture(512, 512, 32*4, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            model = new NNModel();
            model.Load(style.ToString());
            //GetComponent<Camera>().targetTexture = rt;
            GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;

            t2d = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
            rt = new RenderTexture(512, 512, 128);

        }

        void OnDisable()
        {
            model.Release();
        }

     

        void Update()
        {
            GetComponent<Camera>().targetTexture = null;

            if (Input.GetKeyDown(KeyCode.F))
            {

                //将目标摄像机的图像显示到一个板子上
                //pl.GetComponent<Renderer>().material.mainTexture = rt;
                GetComponent<Camera>().targetTexture = rt;
                //截图到t2d中
                RenderTexture.active = rt;

                t2d.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                t2d.Apply();

                RenderTexture.active = null;
                //GameObject.Destroy(rt);
                //getTexture(ref t2d);
                //将图片保存起来
                //byte[] byt = t2d.EncodeToPNG();
                byte[] bytes = t2d.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);

                //string u = string.Format("1{0:D4}", num);

                File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ours.exr", bytes);
                //File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ao.exr", bytes);


                Debug.Log("当前截图序号为：" + num.ToString());
                num++;
            }
        }

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {

            if (mMat == null)
            {
                mMat = new Material(Shader.Find("Hidden/GetInput"));
                mMat.hideFlags = HideFlags.DontSave;
            }


            RenderTexture rtInput = RenderTexture.GetTemporary(src.width, src.height, 32 * 4, 
                RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            Graphics.Blit(src, rtInput, mMat, 0);

            var predict = model.Predict(rtInput);
            RenderTexture.ReleaseTemporary(rtInput);

            Graphics.Blit(predict, dst);
            
        }

    }

    public enum NNStyle
    {
        des_glaneuses,
        la_muse,
        mirror,
        sketch,
        starry_night,
        udnie,
        wave,
        model,
        model_8,
        model_4
    }
}