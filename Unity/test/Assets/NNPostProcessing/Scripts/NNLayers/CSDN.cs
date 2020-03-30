// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class CSDN : MonoBehaviour
{
    private ComputeShader shader;
    private Material mMat;

    int k;
    RenderTexture t;
    void Start()
    {
        GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth 
            | DepthTextureMode.DepthNormals;
        shader = Resources.Load<ComputeShader>("NNLayer");
        k = shader.FindKernel("CSMain");
       
    }


    void OnRenderImage(RenderTexture src, RenderTexture dst)
    {
        if (mMat == null)
        {
            mMat = new Material(Shader.Find("Hidden/ReadDepth"));
            mMat.hideFlags = HideFlags.DontSave;
        }

        int width = src.width;
        int height = src.height;
        var format = RenderTextureFormat.ARGB32;
        var rwMode = RenderTextureReadWrite.Linear;

        var rtMask = RenderTexture.GetTemporary(width, height, 0, format, rwMode);
        Graphics.Blit(src, rtMask, mMat);
        
        t = new RenderTexture(width, height, 24);
        t.enableRandomWrite = true;
        t.Create();

        shader.SetTexture(k, "inputTexture", rtMask);
        shader.SetTexture(k, "Result", t);
        shader.Dispatch(k, src.width/16, src.height/16, 1);
        Graphics.Blit(t, dst);
        t.Release();

        RenderTexture.ReleaseTemporary(rtMask);
    }

}