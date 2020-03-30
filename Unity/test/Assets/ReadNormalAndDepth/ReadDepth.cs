using System.Collections;
using UnityEngine;
using System.IO;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class ReadDepth : MonoBehaviour
{

    //#region Private members

    private Material mMat;
    RenderTexture rt;
    Texture2D t2d;
    int num = 0;
    private bool keydown;

    //#endregion



    #region Built-in resources

    [SerializeField, HideInInspector] Shader _nnaoShader;

    #endregion

    // Start is called before the first frame update
    void Start()
    {
        GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;
        t2d = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        rt = new RenderTexture(512, 512, 32*4,  RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
        //rt = new RenderTexture(new RenderTextureDescriptor(512, 512, RenderTextureFormat.ARGBFloat));

    }


    public void getTexture(ref Texture2D tex)
    {

        for (int y = 0; y < tex.height; y++)
        {
            for (int x = 0; x < tex.width; x++)
            {
                var integer = tex.GetPixel(x, y);
                integer.b += integer.g/255.0f;
                integer.b += integer.r / 65535.0f;
                //integer.g = 1.0f;
                //integer.r = 1.0f;
                tex.SetPixel(x, y, integer);
                //tex
            }
        }

    }

    //void OnPostRender()
    //{
    //    //在每次相机渲染完成时再删除上一帧的texture
    //    GetComponent<Camera>().targetTexture = null;
    //    //设定当前RenderTexture为快照相机的targetTexture
    //    if (Input.GetKeyDown(KeyCode.F))
    //    {
    //        if (t2d != null)
    //        {
    //            Destroy(t2d);
    //        }

    //        RenderTexture rt = GetComponent<Camera>().targetTexture;
    //        RenderTexture.active = rt;
    //        t2d = new Texture2D(rt.width, rt.height);
    //        //读取缓冲区像素信息
    //        t2d.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
    //        t2d.Apply();

    //        byte[] bytes = t2d.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);

    //        //string u = string.Format("1{0:D4}", num);

    //        File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-position.exr", bytes);
    //        //File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ao.exr", bytes);


    //        Debug.Log("当前截图序号为：" + num.ToString());
    //        num++;

    //    }


    //}


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
            //GameObject.Destroy(rt);
            //getTexture(ref t2d);
            //将图片保存起来
            //byte[] byt = t2d.EncodeToPNG();
            byte[] bytes = t2d.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);

            //string u = string.Format("1{0:D4}", num);
            if (num != 0)
            {
                File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-C-Depth.exr", bytes);
                //File.WriteAllBytes("D:\\Projects\\Pycharm\\Ao_pt\\Datastes\\Unity_sceenshot\\test1" + "//" + num.ToString() + "-ao.exr", bytes);
                Debug.Log("当前截图序号为：" + num.ToString());
            }
            var xx = GetComponent<Camera>().projectionMatrix;
            
            //File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ao.exr", bytes);



            num++;
        }
    }


    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {

        if (mMat == null)
        {
            mMat = new Material(Shader.Find("Hidden/ReadDepth"));
            mMat.hideFlags = HideFlags.DontSave;
        }


        Graphics.Blit(source, destination, mMat);

    }
}


