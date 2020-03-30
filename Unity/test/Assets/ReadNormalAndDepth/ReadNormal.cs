using System.Collections;
using UnityEngine;
using System.IO;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class ReadNormal : MonoBehaviour
{


    private Material mMat;
    RenderTexture rt;
    Texture2D t2d;
    int num = 0;

    // Start is called before the first frame update
    void Start()
    {
        GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
        t2d = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        rt = new RenderTexture(512, 512, 32*4, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);

    }

    public void getTexture(ref Texture2D tex)
    {
 
        for (int y = 0; y < tex.height; y++)
        {
            for (int x = 0; x < tex.width; x++)
            {
                var integer = tex.GetPixel(x, y);
                int a = 0;
                //tex
            }
        }

    }

    // Update is called once per frame
    void Update()
    {
        //var xx = GetComponent<Camera>().projectionMatrix;
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
            //var xx = t2d;
            //getTexture(ref t2d);
            //将图片保存起来
            byte[] bytes = t2d.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);

            //string u = string.Format("1{0:D4}", num);
            if (num != 0)
            {
                File.WriteAllBytes("D:\\Projects\\Pycharm\\Ao_pt\\Datastes\\Unity_sceenshot\\test1" + "//" + num.ToString() + "-normal.exr", bytes);
                Debug.Log("当前截图序号为：" + num.ToString());
            }
            num++;
        }
    }


    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {

        if (mMat == null)
        {
            mMat = new Material(Shader.Find("Hidden/ReadNormal"));
            mMat.hideFlags = HideFlags.DontSave;
        }


        Graphics.Blit(source, destination, mMat);

    }
}


