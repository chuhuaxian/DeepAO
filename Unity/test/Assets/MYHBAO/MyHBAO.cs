using System.Collections;
using UnityEngine;
using System.IO;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class MyHBAO : MonoBehaviour
{

    //#region Private members

    private Material mMat;
    RenderTexture rt;
    Texture2D t2d, t2d1;
    int num = 0;
    private bool keydown;

    //#endregion


    // Start is called before the first frame update
    void Start()
    {
        GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
        t2d = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        t2d1 = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        rt = new RenderTexture(512, 512, 32, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        //rt = new RenderTexture(new RenderTextureDescriptor(512, 512, RenderTextureFormat.ARGBFloat));

    }


    public void splitDepthAndNormal(ref Texture2D tex, ref Texture2D tex2)
    {

        for (int y = 0; y < tex.height; y++)
        {
            for (int x = 0; x < tex.width; x++)
            {
                Color integer = tex.GetPixel(x, y);
                integer.r = integer.g = integer.b = integer.a;

                tex2.SetPixel(x, y, integer);
            }
        }

    }



    //// Update is called once per frame
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
            splitDepthAndNormal(ref t2d, ref t2d1);
            //GameObject.Destroy(rt);
            //getTexture(ref t2d);
            //将图片保存起来
            //byte[] byt = t2d.EncodeToPNG();
            byte[] bytes = t2d.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
            byte[] bytes1 = t2d1.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);

            //string u = string.Format("1{0:D4}", num);

            File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-normal.exr", bytes);
            File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-position.exr", bytes1);
            File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ao.exr", bytes);
            //File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ao.exr", bytes);


            Debug.Log("当前截图序号为：" + num.ToString());
            num++;
        }
    }


    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {

        if (mMat == null)
        {
            mMat = new Material(Shader.Find("Hidden/myhbao"));
            mMat.hideFlags = HideFlags.DontSave;
        }


        Graphics.Blit(source, destination, mMat);

    }
}


