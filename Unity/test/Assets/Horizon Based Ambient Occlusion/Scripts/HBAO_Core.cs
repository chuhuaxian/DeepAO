//using UnityEngine;
//using System;
//using System.IO;
//using System.Drawing;


//[AddComponentMenu(null)]
//public class HBAO_Core : MonoBehaviour
//{
//    public enum Preset
//    {
//        FastestPerformance,
//        FastPerformance,
//        Normal,
//        HighQuality,
//        HighestQuality,
//        Custom
//    }

//    public enum IntegrationStage
//    {
//        BeforeImageEffectsOpaque,
//        AfterLighting,
//#if !(UNITY_5_1 || UNITY_5_0)
//        BeforeReflections
//#endif
//    }

//    public enum Quality
//    {
//        Lowest,
//        Low,
//        Medium,
//        High,
//        Highest
//    }

//    public enum Resolution
//    {
//        Full,
//        Half,
//        Quarter
//    }

//    public enum Deinterleaving
//    {
//        Disabled,
//        _2x,
//        _4x
//    }

//    public enum DisplayMode
//    {
//        Normal,
//        AOOnly,
//        ColorBleedingOnly,
//        SplitWithoutAOAndWithAO,
//        SplitWithAOAndAOOnly,
//        SplitWithoutAOAndAOOnly
//    }

//    public enum Blur
//    {
//        None,
//        Narrow,
//        Medium,
//        Wide,
//        ExtraWide
//    }

//    public enum NoiseType
//    {
//        Random,
//        Dither
//    }

//    public enum PerPixelNormals
//    {
//        GBuffer,
//        Camera,
//        Reconstruct
//    }

//    public Texture2D noiseTex;


//    public Texture2D mydepthTex;
//    public Texture2D mynormalTex;


//    public Mesh quadMesh;
//    public Shader hbaoShader = null;

//    [Serializable]
//    public struct Presets
//    {
//        public Preset preset;

//        [SerializeField]
//        public static Presets defaultPresets
//        {
//            get
//            {
//                return new Presets
//                {
//                    preset = Preset.Normal
//                };
//            }
//        }
//    }

//    [Serializable]
//    public struct GeneralSettings
//    {
//        [Tooltip("The stage the AO is integrated into the rendering pipeline.")]
//        [Space(6)]
//        public IntegrationStage integrationStage;

//        [Tooltip("The quality of the AO.")]
//        [Space(10)]
//        public Quality quality;

//        [Tooltip("The deinterleaving factor.")]
//        public Deinterleaving deinterleaving;

//        [Tooltip("The resolution at which the AO is calculated.")]
//        public Resolution resolution;

//        [Tooltip("The type of noise to use.")]
//        [Space(10)]
//        public NoiseType noiseType;

//        [Tooltip("The way the AO is displayed on screen.")]
//        [Space(10)]
//        public DisplayMode displayMode;

//        [SerializeField]
//        public static GeneralSettings defaultSettings
//        {
//            get
//            {
//                return new GeneralSettings
//                {
//                    integrationStage = IntegrationStage.BeforeImageEffectsOpaque,
//                    quality = Quality.Highest,
//                    deinterleaving = Deinterleaving.Disabled,
//                    resolution = Resolution.Full,
//                    noiseType = NoiseType.Dither,
//                    displayMode = DisplayMode.AOOnly
//                };
//            }

//        }
//    }

//    [Serializable]
//    public struct AOSettings
//    {
//        [Tooltip("AO radius: this is the distance outside which occluders are ignored.")]
//        [Space(6), Range(0, 35)]
//        public float radius;

//        [Tooltip("Maximum radius in pixels: this prevents the radius to grow too much with close-up " +
//                  "object and impact on performances.")]
//        [Range(64, 512)]
//        public float maxRadiusPixels;

//        [Tooltip("For low-tessellated geometry, occlusion variations tend to appear at creases and " +
//                 "ridges, which betray the underlying tessellation. To remove these artifacts, we use " +
//                 "an angle bias parameter which restricts the hemisphere.")]
//        [Range(0, 0.5f)]
//        public float bias;

//        [Tooltip("This value allows to scale up the ambient occlusion values.")]
//        [Range(0, 10)]
//        public float intensity;

//        [Tooltip("Enable/disable MultiBounce approximation.")]
//        public bool useMultiBounce;

//        [Tooltip("MultiBounce approximation influence.")]
//        [Range(0, 1)]
//        public float multiBounceInfluence;

//        [Tooltip("The amount of AO offscreen samples are contributing.")]
//        [Range(0, 1)]
//        public float offscreenSamplesContribution;

//        [Tooltip("The max distance to display AO.")]
//        [Space(10)]
//        public float maxDistance;

//        [Tooltip("The distance before max distance at which AO start to decrease.")]
//        public float distanceFalloff;

//        [Tooltip("The type of per pixel normals to use.")]
//        [Space(10)]
//        public PerPixelNormals perPixelNormals;

//        [Tooltip("This setting allow you to set the base color if the AO, the alpha channel value is unused.")]
//        [Space(10)]
//        public Color baseColor;

//        [SerializeField]
//        public static AOSettings defaultSettings
//        {
//            get
//            {
//                return new AOSettings
//                {
//                    radius = 25.0f,
//                    maxRadiusPixels = 128f,
//                    bias = 0.05f,
//                    intensity = 1f,
//                    useMultiBounce = false,
//                    multiBounceInfluence = 1f,
//                    offscreenSamplesContribution = 0f,
//                    maxDistance = 1500f,
//                    distanceFalloff = 0f,
//                    perPixelNormals = PerPixelNormals.GBuffer,
//                    baseColor = Color.black
//                };
//            }

//        }

//        public void set_radius(float r)
//        {
//            radius = r;
//        }

//        public float get_radius()
//        {
//            return radius;
//        }

//    }

//    [Serializable]
//    public struct ColorBleedingSettings
//    {
//        [Space(6)]
//        public bool enabled;

//        [Tooltip("This value allows to control the saturation of the color bleeding.")]
//        [Space(10), Range(0, 4)]
//        public float saturation;

//        [Tooltip("This value allows to scale the contribution of the color bleeding samples.")]
//        [Range(0, 32)]
//        public float albedoMultiplier;

//        [Tooltip("Use masking on emissive pixels")]
//        [Range(0, 1)]
//        public float brightnessMask;

//        [Tooltip("Brightness level where masking starts/ends")]
//        [HBAO_MinMaxSlider(0, 8)]
//        public Vector2 brightnessMaskRange;

//        [SerializeField]
//        public static ColorBleedingSettings defaultSettings
//        {
//            get
//            {
//                return new ColorBleedingSettings
//                {
//                    enabled = false,
//                    saturation = 1f,
//                    albedoMultiplier = 4f,
//                    brightnessMask = 1f,
//                    brightnessMaskRange = new Vector2(0.8f, 1.2f)
//                };
//            }
//        }
//    }

//    [Serializable]
//    public struct BlurSettings
//    {
//        [Tooltip("The type of blur to use.")]
//        [Space(6)]
//        public Blur amount;

//        [Tooltip("This parameter controls the depth-dependent weight of the bilateral filter, to " +
//                 "avoid bleeding across edges. A zero sharpness is a pure Gaussian blur. Increasing " +
//                 "the blur sharpness removes bleeding by using lower weights for samples with large " +
//                 "depth delta from the current pixel.")]
//        [Space(10), Range(0, 16)]
//        public float sharpness;

//        [Tooltip("Is the blur downsampled.")]
//        public bool downsample;

//        [SerializeField]
//        public static BlurSettings defaultSettings
//        {
//            get
//            {
//                return new BlurSettings
//                {
//                    amount = Blur.Medium,
//                    sharpness = 8f,
//                    downsample = false
//                };
//            }
//        }
//    }

//    [AttributeUsage(AttributeTargets.Field)]
//    public class SettingsGroup : Attribute { }

//    [SerializeField, SettingsGroup]
//    private Presets m_Presets = Presets.defaultPresets;
//    public Presets presets
//    {
//        get { return m_Presets; }
//        set { m_Presets = value; }
//    }

//    [SerializeField, SettingsGroup]
//    private GeneralSettings m_GeneralSettings = GeneralSettings.defaultSettings;
//    public GeneralSettings generalSettings
//    {
//        get { return m_GeneralSettings; }
//        set { m_GeneralSettings = value; }
//    }

//    [SerializeField, SettingsGroup]
//    private AOSettings m_AOSettings = AOSettings.defaultSettings;
//    public AOSettings aoSettings
//    {
//        get { return m_AOSettings; }
//        set { m_AOSettings = value; }
//    }

//    [SerializeField, SettingsGroup]
//    private ColorBleedingSettings m_ColorBleedingSettings = ColorBleedingSettings.defaultSettings;
//    public ColorBleedingSettings colorBleedingSettings
//    {
//        get { return m_ColorBleedingSettings; }
//        set { m_ColorBleedingSettings = value; }
//    }

//    [SerializeField, SettingsGroup]
//    private BlurSettings m_BlurSettings = BlurSettings.defaultSettings;
//    public BlurSettings blurSettings
//    {
//        get { return m_BlurSettings; }
//        set { m_BlurSettings = value; }
//    }

//    private static class MersenneTwister
//    {
//        // Mersenne-Twister random numbers in [0,1).
//        public static float[] Numbers = new float[] {
//            0.463937f,0.340042f,0.223035f,0.468465f,0.322224f,0.979269f,0.031798f,0.973392f,0.778313f,0.456168f,0.258593f,0.330083f,0.387332f,0.380117f,0.179842f,0.910755f,
//            0.511623f,0.092933f,0.180794f,0.620153f,0.101348f,0.556342f,0.642479f,0.442008f,0.215115f,0.475218f,0.157357f,0.568868f,0.501241f,0.629229f,0.699218f,0.707733f
//        };
//    }

//    protected static class Pass
//    {
//        public const int AO_LowestQuality = 0;
//        public const int AO_LowQuality = 1;
//        public const int AO_MediumQuality = 2;
//        public const int AO_HighQuality = 3;
//        public const int AO_HighestQuality = 4;
//        public const int AO_Deinterleaved_LowestQuality = 5;
//        public const int AO_Deinterleaved_LowQuality = 6;
//        public const int AO_Deinterleaved_MediumQuality = 7;
//        public const int AO_Deinterleaved_HighQuality = 8;
//        public const int AO_Deinterleaved_HighestQuality = 9;

//        public const int Depth_Deinterleaving_2x2 = 10;
//        public const int Depth_Deinterleaving_4x4 = 11;
//        public const int Normals_Deinterleaving_2x2 = 12;
//        public const int Normals_Deinterleaving_4x4 = 13;

//        public const int Atlas = 14;

//        public const int Reinterleaving_2x2 = 15;
//        public const int Reinterleaving_4x4 = 16;

//        public const int Blur_X_Narrow = 17;
//        public const int Blur_X_Medium = 18;
//        public const int Blur_X_Wide = 19;
//        public const int Blur_X_ExtraWide = 20;
//        public const int Blur_Y_Narrow = 21;
//        public const int Blur_Y_Medium = 22;
//        public const int Blur_Y_Wide = 23;
//        public const int Blur_Y_ExtraWide = 24;

//        public const int Composite = 25;
//        public const int Composite_MultiBounce = 26;
//        public const int Debug_AO_Only = 27;
//        public const int Debug_AO_Only_MultiBounce = 28;
//        public const int Debug_ColorBleeding_Only = 29;
//        public const int Debug_Split_WithoutAO_WithAO = 30;
//        public const int Debug_Split_WithoutAO_WithAO_MultiBounce = 31;
//        public const int Debug_Split_WithAO_AOOnly = 32;
//        public const int Debug_Split_WithAO_AOOnly_MultiBounce = 33;
//        public const int Debug_Split_WithoutAO_AOOnly = 34;
//        public const int Debug_Split_WithoutAO_AOOnly_MultiBounce = 35;

//        public const int Combine_Deffered = 36;
//        public const int Combine_Deffered_Multiplicative = 37;
//        public const int Combine_Integrated = 38;
//        public const int Combine_Integrated_MultiBounce = 39;
//        public const int Combine_Integrated_Multiplicative = 40;
//        public const int Combine_Integrated_Multiplicative_MultiBounce = 41;
//        public const int Combine_ColorBleeding = 42;
//        public const int Debug_Split_Additive = 43;
//        public const int Debug_Split_Additive_MultiBounce = 44;
//        public const int Debug_Split_Multiplicative = 45;
//        public const int Debug_Split_Multiplicative_MultiBounce = 46;
//    }

//    protected class RenderTarget
//    {
//        public bool orthographic;
//        public RenderingPath renderingPath;
//        public bool hdr;
//        public int width;
//        public int height;
//        public int fullWidth;
//        public int fullHeight;
//        public int layerWidth;
//        public int layerHeight;
//        public int downsamplingFactor;
//        public int deinterleavingFactor;
//        public int blurDownsamplingFactor;
//    }

//    protected static class ShaderProperties
//    {
//        public static int mainTex;
//        public static int hbaoTex;
//        public static int noiseTex;


//        public static int mydepthTex;
//        public static int mynormalTex;


//        public static int rt0Tex;
//        public static int rt3Tex;
//        public static int depthTex;
//        public static int normalsTex;
//        public static int[] mrtDepthTex;
//        public static int[] mrtNrmTex;
//        public static int[] mrtHBAOTex;
//        public static int[] deinterleavingOffset;
//        public static int layerOffset;
//        public static int jitter;
//        public static int uvToView;
//        public static int worldToCameraMatrix;
//        public static int fullResTexelSize;
//        public static int layerResTexelSize;
//        public static int targetScale;
//        public static int noiseTexSize;
//        public static int radius;
//        public static int maxRadiusPixels;
//        public static int negInvRadius2;
//        public static int angleBias;
//        public static int aoMultiplier;
//        public static int intensity;
//        public static int multiBounceInfluence;
//        public static int offscreenSamplesContrib;
//        public static int maxDistance;
//        public static int distanceFalloff;
//        public static int baseColor;
//        public static int colorBleedSaturation;
//        public static int albedoMultiplier;
//        public static int colorBleedBrightnessMask;
//        public static int colorBleedBrightnessMaskRange;
//        public static int blurSharpness;


//        static ShaderProperties()
//        {
//            mainTex = Shader.PropertyToID("_MainTex");
//            hbaoTex = Shader.PropertyToID("_HBAOTex");

//            mydepthTex = Shader.PropertyToID("_MyDepthTex");
//            mynormalTex = Shader.PropertyToID("_MyNormalTex");

//            noiseTex = Shader.PropertyToID("_NoiseTex");
//            rt0Tex = Shader.PropertyToID("_rt0Tex");
//            rt3Tex = Shader.PropertyToID("_rt3Tex");
//            depthTex = Shader.PropertyToID("_DepthTex");
//            normalsTex = Shader.PropertyToID("_NormalsTex");  //-------------------NOAMAL---------------------//
//            mrtDepthTex = new int[4 * NUM_MRTS];
//            mrtNrmTex = new int[4 * NUM_MRTS];
//            mrtHBAOTex = new int[4 * NUM_MRTS];
//            for (int i = 0; i < 4 * NUM_MRTS; i++)
//            {
//                mrtDepthTex[i] = Shader.PropertyToID("_DepthLayerTex" + i);
//                mrtNrmTex[i] = Shader.PropertyToID("_NormalLayerTex" + i);
//                mrtHBAOTex[i] = Shader.PropertyToID("_HBAOLayerTex" + i);
//            }

//            deinterleavingOffset = new int[] {
//                Shader.PropertyToID("_Deinterleaving_Offset00"),
//                Shader.PropertyToID("_Deinterleaving_Offset10"),
//                Shader.PropertyToID("_Deinterleaving_Offset01"),
//                Shader.PropertyToID("_Deinterleaving_Offset11")
//            };

//            layerOffset = Shader.PropertyToID("_LayerOffset");
//            jitter = Shader.PropertyToID("_Jitter");
//            uvToView = Shader.PropertyToID("_UVToView");
//            worldToCameraMatrix = Shader.PropertyToID("_WorldToCameraMatrix");
//            fullResTexelSize = Shader.PropertyToID("_FullRes_TexelSize");
//            layerResTexelSize = Shader.PropertyToID("_LayerRes_TexelSize");
//            targetScale = Shader.PropertyToID("_TargetScale");
//            noiseTexSize = Shader.PropertyToID("_NoiseTexSize");
//            radius = Shader.PropertyToID("_Radius");
//            maxRadiusPixels = Shader.PropertyToID("_MaxRadiusPixels");
//            negInvRadius2 = Shader.PropertyToID("_NegInvRadius2");
//            angleBias = Shader.PropertyToID("_AngleBias");
//            aoMultiplier = Shader.PropertyToID("_AOmultiplier");
//            intensity = Shader.PropertyToID("_Intensity");
//            multiBounceInfluence = Shader.PropertyToID("_MultiBounceInfluence");
//            offscreenSamplesContrib = Shader.PropertyToID("_OffscreenSamplesContrib");
//            maxDistance = Shader.PropertyToID("_MaxDistance");
//            distanceFalloff = Shader.PropertyToID("_DistanceFalloff");
//            baseColor = Shader.PropertyToID("_BaseColor");
//            colorBleedSaturation = Shader.PropertyToID("_ColorBleedSaturation");
//            albedoMultiplier = Shader.PropertyToID("_AlbedoMultiplier");
//            colorBleedBrightnessMask = Shader.PropertyToID("_ColorBleedBrightnessMask");
//            colorBleedBrightnessMaskRange = Shader.PropertyToID("_ColorBleedBrightnessMaskRange");
//            blurSharpness = Shader.PropertyToID("_BlurSharpness");
//        }
//    }

//    protected Material _hbaoMaterial;
//    protected Camera _hbaoCamera;
//    protected RenderTarget _renderTarget;
//    protected const int NUM_MRTS = 4;
//    protected const int NUM_SANPLES = 100;

//    protected Vector4[] _jitter = new Vector4[4 * NUM_MRTS];

//    private Quality _quality;
//    private NoiseType _noiseType;
//    private string[] _hbaoShaderKeywords = new string[4];
//    private int[] _numSampleDirections = new int[] { 3, 4, 6, 8, 8 }; // LOWEST, LOW, MEDIUM, HIGH, HIGHEST (highest uses more steps)

//    protected virtual void OnEnable()
//    {
//        if (!SystemInfo.supportsImageEffects || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.Depth))
//        {
//            Debug.LogWarning("HBAO shader is not supported on this platform.");
//            this.enabled = false;
//            return;
//        }

//        if (hbaoShader != null && !hbaoShader.isSupported)
//        {
//            Debug.LogWarning("HBAO shader is not supported on this platform.");
//            this.enabled = false;
//            return;
//        }

//        if (hbaoShader == null)
//        {
//            return;
//        }

//        CreateMaterial();

//        _hbaoCamera.depthTextureMode |= DepthTextureMode.Depth;
//        if (aoSettings.perPixelNormals == PerPixelNormals.Camera)
//            _hbaoCamera.depthTextureMode |= DepthTextureMode.DepthNormals;
//#if UNITY_5_6_OR_NEWER
//        _hbaoCamera.forceIntoRenderTexture = true;
//#endif
//    }

//    protected virtual void OnDisable()
//    {
//        if (_hbaoMaterial != null)
//            DestroyImmediate(_hbaoMaterial);
//        if (noiseTex != null)
//            DestroyImmediate(noiseTex);


//        if (mynormalTex != null)
//            DestroyImmediate(mynormalTex);
//        if (mydepthTex != null)
//            DestroyImmediate(mydepthTex);


//        if (quadMesh != null)
//            DestroyImmediate(quadMesh);
//    }

//    private void CreateMaterial()
//    {
//        if (_hbaoMaterial == null)
//        {
//            _hbaoMaterial = new Material(hbaoShader);
//            _hbaoMaterial.hideFlags = HideFlags.HideAndDontSave;

//            _hbaoCamera = GetComponent<Camera>();
//        }

//        if (quadMesh != null)
//            DestroyImmediate(quadMesh);

//        quadMesh = new Mesh();
//        quadMesh.vertices = new Vector3[] {
//            new Vector3(-0.5f, -0.5f, 0),
//            new Vector3( 0.5f,  0.5f, 0),
//            new Vector3( 0.5f, -0.5f, 0),
//            new Vector3(-0.5f,  0.5f, 0)
//        };
//        quadMesh.uv = new Vector2[] {
//            new Vector2(0, 0),
//            new Vector2(1, 1),
//            new Vector2(1, 0),
//            new Vector2(0, 1)
//        };
//        quadMesh.triangles = new int[] { 0, 1, 2, 1, 0, 3 };

//        _renderTarget = new RenderTarget();
//    }

//    //public static Texture2D GetTexrture2DFromPath(string imgPath)
//    //{
//    //    //读取文件
//    //    FileStream fs = new FileStream(imgPath, FileMode.Open, FileAccess.Read);
//    //    int byteLength = (int)fs.Length;
//    //    byte[] imgBytes = new byte[byteLength];
//    //    fs.Read(imgBytes, 0, byteLength);
//    //    fs.Close();
//    //    fs.Dispose();
//    //    //转化为Texture2D
//    //    Image img = Image.FromStream(new MemoryStream(imgBytes));
//    //    Texture2D t2d = new Texture2D(img.Width, img.Height);
//    //    img.Dispose();
//    //    t2d.LoadImage(imgBytes);
//    //    t2d.Apply();
//    //    return t2d;
//    //}


//    protected void UpdateShaderProperties(int num)
//    {
//        _renderTarget.orthographic = _hbaoCamera.orthographic;
//        _renderTarget.renderingPath = _hbaoCamera.actualRenderingPath;
//#if UNITY_5_6_OR_NEWER
//        _renderTarget.hdr = _hbaoCamera.allowHDR;
//#else
//        _renderTarget.hdr = _hbaoCamera.hdr;
//#endif
//#if UNITY_2017_2_OR_NEWER
//        if (UnityEngine.XR.XRSettings.enabled)
//        {
//            _renderTarget.width = UnityEngine.XR.XRSettings.eyeTextureDesc.width;
//            _renderTarget.height = UnityEngine.XR.XRSettings.eyeTextureDesc.height;
//        }
//        else
//        {
//            _renderTarget.width = _hbaoCamera.pixelWidth;
//            _renderTarget.height = _hbaoCamera.pixelHeight;
//        }
//#else
//        _renderTarget.width = _hbaoCamera.pixelWidth;
//        _renderTarget.height = _hbaoCamera.pixelHeight;
//#endif
//        _renderTarget.downsamplingFactor = generalSettings.resolution == Resolution.Full ? 1 : generalSettings.resolution == Resolution.Half ? 2 : 4;
//        _renderTarget.deinterleavingFactor = GetDeinterleavingFactor();
//        _renderTarget.blurDownsamplingFactor = blurSettings.downsample ? 2 : 1;

//        //float tanHalfFovY = Mathf.Tan(0.5f * _hbaoCamera.fieldOfView * Mathf.Deg2Rad);
//        //float invFocalLenX = 1.0f / (1.0f / tanHalfFovY * (_renderTarget.height / (float)_renderTarget.width));
//        //float invFocalLenY = 1.0f / (1.0f / tanHalfFovY);
//        //_hbaoMaterial.SetVector(ShaderProperties.uvToView, new Vector4(2.0f * invFocalLenX, -2.0f * invFocalLenY, -1.0f * invFocalLenX, 1.0f * invFocalLenY));

//        float tanHalfFovY = 0.4663077f;
//        _hbaoMaterial.SetVector(ShaderProperties.uvToView, new Vector4(2.0f * tanHalfFovY, -2.0f * tanHalfFovY, -1.0f * tanHalfFovY, 1.0f * tanHalfFovY));


//        _hbaoMaterial.SetMatrix(ShaderProperties.worldToCameraMatrix, _hbaoCamera.worldToCameraMatrix);

//        if (generalSettings.deinterleaving != Deinterleaving.Disabled)
//        {
//            _renderTarget.fullWidth = _renderTarget.width + (_renderTarget.width % _renderTarget.deinterleavingFactor == 0 ? 0 : _renderTarget.deinterleavingFactor - (_renderTarget.width % _renderTarget.deinterleavingFactor));
//            _renderTarget.fullHeight = _renderTarget.height + (_renderTarget.height % _renderTarget.deinterleavingFactor == 0 ? 0 : _renderTarget.deinterleavingFactor - (_renderTarget.height % _renderTarget.deinterleavingFactor));
//            _renderTarget.layerWidth = _renderTarget.fullWidth / _renderTarget.deinterleavingFactor;
//            _renderTarget.layerHeight = _renderTarget.fullHeight / _renderTarget.deinterleavingFactor;

//            _hbaoMaterial.SetVector(ShaderProperties.fullResTexelSize, new Vector4(1.0f / _renderTarget.fullWidth, 1.0f / _renderTarget.fullHeight, _renderTarget.fullWidth, _renderTarget.fullHeight));
//            _hbaoMaterial.SetVector(ShaderProperties.layerResTexelSize, new Vector4(1.0f / _renderTarget.layerWidth, 1.0f / _renderTarget.layerHeight, _renderTarget.layerWidth, _renderTarget.layerHeight));
//            _hbaoMaterial.SetVector(ShaderProperties.targetScale, new Vector4(_renderTarget.fullWidth / (float)_renderTarget.width, _renderTarget.fullHeight / (float)_renderTarget.height, 1.0f / (_renderTarget.fullWidth / (float)_renderTarget.width), 1.0f / (_renderTarget.fullHeight / (float)_renderTarget.height)));
//        }
//        else
//        {
//            _renderTarget.fullWidth = _renderTarget.width;
//            _renderTarget.fullHeight = _renderTarget.height;
//            if (generalSettings.resolution == Resolution.Half && aoSettings.perPixelNormals == PerPixelNormals.Reconstruct)
//                _hbaoMaterial.SetVector(ShaderProperties.targetScale, new Vector4((_renderTarget.width + 0.5f) / _renderTarget.width, (_renderTarget.height + 0.5f) / _renderTarget.height, 1f, 1f));
//            else
//                _hbaoMaterial.SetVector(ShaderProperties.targetScale, new Vector4(1f, 1f, 1f, 1f));
//        }

//        if (noiseTex == null || _quality != generalSettings.quality || _noiseType != generalSettings.noiseType)
//        {
//            if (noiseTex != null)
//                DestroyImmediate(noiseTex);

//            float noiseTexSize = generalSettings.noiseType == NoiseType.Dither ? 4 : 64;
//            CreateRandomTexture((int)noiseTexSize);
//        }

//        //if (mydepthnormalTex == null)
//        //{
//        //    mydepthnormalTex = new Texture2D(512, 512, TextureFormat.RGBAFloat, false, true);
//        //    byte[] fileData;
//        //    string filePath = "D:\\Projects\\Unity\\test_03\\screenshot\\2-C-Depth.exr";
//        //    fileData = File.ReadAllBytes(filePath);
//        //    mydepthnormalTex.LoadImage(fileData);
//        //    mydepthnormalTex.Apply();
//        //}


//        mydepthTex = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
//        byte[] fileData;
//        string filePath = "D:\\Projects\\Unity\\test_03\\screenshot\\depth-3.png";
//        fileData = File.ReadAllBytes(filePath);

//        mydepthTex.LoadImage(fileData);
//        mydepthTex.Apply();

//        mynormalTex = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
//        byte[] fileData_normal;
//        //int num = 2;
//        string filePath_normal = "D:\\Projects\\Unity\\test_03\\screenshot\\normal-3.png";
//        //string filePath_normal = "D:\\Projects\\Unity\\test_03\\screenshot\\normal-" + num.ToString() + ".png";
//        fileData_normal = File.ReadAllBytes(filePath_normal);

//        mynormalTex.LoadImage(fileData_normal);
//        mynormalTex.Apply();



//        _quality = generalSettings.quality;
//        _noiseType = generalSettings.noiseType;

//        _hbaoMaterial.SetTexture(ShaderProperties.noiseTex, noiseTex);

//        _hbaoMaterial.SetTexture(ShaderProperties.mydepthTex, mydepthTex);
//        _hbaoMaterial.SetTexture(ShaderProperties.mynormalTex, mynormalTex);

//        _hbaoMaterial.SetFloat(ShaderProperties.noiseTexSize, _noiseType == NoiseType.Dither ? 4 : 64);
//        _hbaoMaterial.SetFloat(ShaderProperties.radius, aoSettings.radius * 0.5f * (_renderTarget.height / (tanHalfFovY * 2.0f)) / _renderTarget.deinterleavingFactor);
//        _hbaoMaterial.SetFloat(ShaderProperties.maxRadiusPixels, aoSettings.maxRadiusPixels / _renderTarget.deinterleavingFactor);
//        _hbaoMaterial.SetFloat(ShaderProperties.negInvRadius2, -1.0f / (aoSettings.radius * aoSettings.radius));
//        _hbaoMaterial.SetFloat(ShaderProperties.angleBias, aoSettings.bias);
//        _hbaoMaterial.SetFloat(ShaderProperties.aoMultiplier, 2.0f * (1.0f / (1.0f - aoSettings.bias)));
//        _hbaoMaterial.SetFloat(ShaderProperties.intensity, aoSettings.intensity);
//        _hbaoMaterial.SetFloat(ShaderProperties.multiBounceInfluence, aoSettings.multiBounceInfluence);
//        _hbaoMaterial.SetFloat(ShaderProperties.offscreenSamplesContrib, aoSettings.offscreenSamplesContribution);
//        _hbaoMaterial.SetFloat(ShaderProperties.maxDistance, aoSettings.maxDistance);
//        _hbaoMaterial.SetFloat(ShaderProperties.distanceFalloff, aoSettings.distanceFalloff);
//        _hbaoMaterial.SetColor(ShaderProperties.baseColor, aoSettings.baseColor);
//        _hbaoMaterial.SetFloat(ShaderProperties.colorBleedSaturation, colorBleedingSettings.saturation);
//        _hbaoMaterial.SetFloat(ShaderProperties.albedoMultiplier, colorBleedingSettings.albedoMultiplier);
//        _hbaoMaterial.SetFloat(ShaderProperties.colorBleedBrightnessMask, colorBleedingSettings.brightnessMask);
//        _hbaoMaterial.SetVector(ShaderProperties.colorBleedBrightnessMaskRange, colorBleedingSettings.brightnessMaskRange);
//        _hbaoMaterial.SetFloat(ShaderProperties.blurSharpness, blurSettings.sharpness);
//    }

//    protected void UpdateShaderKeywords()
//    {
//        _hbaoShaderKeywords[0] = colorBleedingSettings.enabled ? "COLOR_BLEEDING_ON" : "__";

//        if (_renderTarget.orthographic)
//            _hbaoShaderKeywords[1] = "ORTHOGRAPHIC_PROJECTION_ON";
//        else
//            _hbaoShaderKeywords[1] = IsDeferredShading() ? "DEFERRED_SHADING_ON" : "__";

//        _hbaoShaderKeywords[2] = aoSettings.perPixelNormals == PerPixelNormals.Camera ? "NORMALS_CAMERA" : aoSettings.perPixelNormals == PerPixelNormals.Reconstruct ? "NORMALS_RECONSTRUCT" : "__";

//        _hbaoShaderKeywords[3] = aoSettings.offscreenSamplesContribution > 0 ? "OFFSCREEN_SAMPLES_CONTRIB" : "__";

//        _hbaoMaterial.shaderKeywords = _hbaoShaderKeywords;
//    }

//    protected virtual void CheckParameters()
//    {
//        if (!IsDeferredShading() && aoSettings.perPixelNormals == PerPixelNormals.GBuffer)
//            m_AOSettings.perPixelNormals = PerPixelNormals.Camera;

//        if (generalSettings.deinterleaving != Deinterleaving.Disabled && SystemInfo.supportedRenderTargetCount < 4)
//            m_GeneralSettings.deinterleaving = Deinterleaving.Disabled;
//    }

//    protected bool IsDeferredShading()
//    {
//        return _hbaoCamera.actualRenderingPath == RenderingPath.DeferredShading;
//    }

//    protected bool IsDeferredShadingOrLighting()
//    {
//        return _hbaoCamera.actualRenderingPath == RenderingPath.DeferredShading || _hbaoCamera.actualRenderingPath == RenderingPath.DeferredLighting;
//    }

//    protected int GetDeinterleavingFactor()
//    {
//        switch (generalSettings.deinterleaving)
//        {
//            case Deinterleaving._2x:
//                return 2;
//            case Deinterleaving._4x:
//                return 4;
//            case Deinterleaving.Disabled:
//            default:
//                return 1;
//        }
//    }

//    protected int GetAoPass()
//    {
//        switch (generalSettings.quality)
//        {
//            case Quality.Lowest:
//                return Pass.AO_LowestQuality;
//            case Quality.Low:
//                return Pass.AO_LowQuality;
//            case Quality.Medium:
//                return Pass.AO_MediumQuality;
//            case Quality.High:
//                return Pass.AO_HighQuality;
//            case Quality.Highest:
//                return Pass.AO_HighestQuality;
//            default:
//                return Pass.AO_MediumQuality;
//        }
//    }

//    protected int GetAoDeinterleavedPass()
//    {
//        switch (generalSettings.quality)
//        {
//            case Quality.Lowest:
//                return Pass.AO_Deinterleaved_LowestQuality;
//            case Quality.Low:
//                return Pass.AO_Deinterleaved_LowQuality;
//            case Quality.Medium:
//                return Pass.AO_Deinterleaved_MediumQuality;
//            case Quality.High:
//                return Pass.AO_Deinterleaved_HighQuality;
//            case Quality.Highest:
//                return Pass.AO_Deinterleaved_HighestQuality;
//            default:
//                return Pass.AO_Deinterleaved_MediumQuality;
//        }
//    }

//    protected int GetBlurXPass()
//    {
//        switch (blurSettings.amount)
//        {
//            case Blur.Narrow:
//                return Pass.Blur_X_Narrow;
//            case Blur.Medium:
//                return Pass.Blur_X_Medium;
//            case Blur.Wide:
//                return Pass.Blur_X_Wide;
//            case Blur.ExtraWide:
//                return Pass.Blur_X_ExtraWide;
//            default:
//                return Pass.Blur_X_Medium;
//        }
//    }

//    protected int GetBlurYPass()
//    {
//        switch (blurSettings.amount)
//        {
//            case Blur.Narrow:
//                return Pass.Blur_Y_Narrow;
//            case Blur.Medium:
//                return Pass.Blur_Y_Medium;
//            case Blur.Wide:
//                return Pass.Blur_Y_Wide;
//            case Blur.ExtraWide:
//                return Pass.Blur_Y_ExtraWide;
//            default:
//                return Pass.Blur_Y_Medium;
//        }
//    }

//    protected int GetFinalPass()
//    {
//        switch (generalSettings.displayMode)
//        {
//            case DisplayMode.Normal:
//                return aoSettings.useMultiBounce ? Pass.Composite_MultiBounce : Pass.Composite;
//            case DisplayMode.AOOnly:
//                return aoSettings.useMultiBounce ? Pass.Debug_AO_Only_MultiBounce : Pass.Debug_AO_Only;
//            case DisplayMode.ColorBleedingOnly:
//                return Pass.Debug_ColorBleeding_Only;
//            case DisplayMode.SplitWithoutAOAndWithAO:
//                return aoSettings.useMultiBounce ? Pass.Debug_Split_WithoutAO_WithAO_MultiBounce : Pass.Debug_Split_WithoutAO_WithAO;
//            case DisplayMode.SplitWithAOAndAOOnly:
//                return aoSettings.useMultiBounce ? Pass.Debug_Split_WithAO_AOOnly_MultiBounce : Pass.Debug_Split_WithAO_AOOnly;
//            case DisplayMode.SplitWithoutAOAndAOOnly:
//                return aoSettings.useMultiBounce ? Pass.Debug_Split_WithoutAO_AOOnly_MultiBounce : Pass.Debug_Split_WithoutAO_AOOnly;
//            default:
//                return Pass.Composite;
//        }
//    }

//    private void CreateRandomTexture(int size)
//    {
//        noiseTex = new Texture2D(size, size, TextureFormat.RGB24, false, true);
//        noiseTex.filterMode = FilterMode.Point;
//        noiseTex.wrapMode = TextureWrapMode.Repeat;
//        int z = 0;
//        for (int x = 0; x < size; ++x)
//        {
//            for (int y = 0; y < size; ++y)
//            {
//                float r1 = generalSettings.noiseType == NoiseType.Dither ? MersenneTwister.Numbers[z++] : UnityEngine.Random.Range(0.0f, 1.0f);
//                float r2 = generalSettings.noiseType == NoiseType.Dither ? MersenneTwister.Numbers[z++] : UnityEngine.Random.Range(0.0f, 1.0f);
//                float angle = 2.0f * Mathf.PI * r1 / _numSampleDirections[GetAoPass()];
//                Color color = new Color(Mathf.Cos(angle), Mathf.Sin(angle), r2);
//                noiseTex.SetPixel(x, y, color);
//            }
//        }
//        noiseTex.Apply();

//        for (int i = 0, j = 0; i < _jitter.Length; ++i)
//        {
//            float r1 = MersenneTwister.Numbers[j++];
//            float r2 = MersenneTwister.Numbers[j++];
//            float angle = 2.0f * Mathf.PI * r1 / _numSampleDirections[GetAoPass()];
//            _jitter[i] = new Vector4(Mathf.Cos(angle), Mathf.Sin(angle), r2, 0);
//        }
//    }

//    public void ApplyPreset(Preset preset)
//    {
//        if (preset == Preset.Custom)
//        {
//            m_Presets.preset = preset;
//            return;
//        }

//        DisplayMode displayMode = generalSettings.displayMode;

//        m_GeneralSettings = GeneralSettings.defaultSettings;
//        m_AOSettings = AOSettings.defaultSettings;
//        m_ColorBleedingSettings = ColorBleedingSettings.defaultSettings;
//        m_BlurSettings = BlurSettings.defaultSettings;

//        m_GeneralSettings.displayMode = displayMode;

//        switch (preset)
//        {
//            case Preset.FastestPerformance:
//                m_GeneralSettings.quality = Quality.Lowest;
//                m_AOSettings.radius = 0.5f;
//                m_AOSettings.maxRadiusPixels = 64.0f;
//                m_BlurSettings.amount = Blur.ExtraWide;
//                m_BlurSettings.downsample = true;
//                break;
//            case Preset.FastPerformance:
//                m_GeneralSettings.quality = Quality.Low;
//                m_AOSettings.radius = 0.5f;
//                m_AOSettings.maxRadiusPixels = 64.0f;
//                m_BlurSettings.amount = Blur.Wide;
//                m_BlurSettings.downsample = true;
//                break;
//            case Preset.HighQuality:
//                m_GeneralSettings.quality = Quality.High;
//                m_AOSettings.radius = 1.0f;
//                break;
//            case Preset.HighestQuality:
//                m_GeneralSettings.quality = Quality.Highest;
//                m_AOSettings.radius = 1.2f;
//                m_AOSettings.maxRadiusPixels = 256.0f;
//                m_BlurSettings.amount = Blur.Narrow;
//                break;
//            case Preset.Normal:
//            default:
//                break;
//        }

//        m_Presets.preset = preset;
//    }
//}



using UnityEngine;
using System;
using System.IO;
using System.Drawing;


[AddComponentMenu(null)]
public class HBAO_Core : MonoBehaviour
{
    public enum Preset
    {
        FastestPerformance,
        FastPerformance,
        Normal,
        HighQuality,
        HighestQuality,
        Custom
    }

    public enum IntegrationStage
    {
        BeforeImageEffectsOpaque,
        AfterLighting,
#if !(UNITY_5_1 || UNITY_5_0)
        BeforeReflections
#endif
    }

    public enum Quality
    {
        Lowest,
        Low,
        Medium,
        High,
        Highest
    }

    public enum Resolution
    {
        Full,
        Half,
        Quarter
    }

    public enum Deinterleaving
    {
        Disabled,
        _2x,
        _4x
    }

    public enum DisplayMode
    {
        Normal,
        AOOnly,
        ColorBleedingOnly,
        SplitWithoutAOAndWithAO,
        SplitWithAOAndAOOnly,
        SplitWithoutAOAndAOOnly
    }

    public enum Blur
    {
        None,
        Narrow,
        Medium,
        Wide,
        ExtraWide
    }

    public enum NoiseType
    {
        Random,
        Dither
    }

    public enum PerPixelNormals
    {
        GBuffer,
        Camera,
        Reconstruct
    }

    public Texture2D noiseTex;


    public Texture2D mydepthTex;
    public Texture2D mynormalTex;


    public Mesh quadMesh;
    public Shader hbaoShader = null;

    [Serializable]
    public struct Presets
    {
        public Preset preset;

        [SerializeField]
        public static Presets defaultPresets
        {
            get
            {
                return new Presets
                {
                    preset = Preset.Normal
                };
            }
        }
    }

    [Serializable]
    public struct GeneralSettings
    {
        [Tooltip("The stage the AO is integrated into the rendering pipeline.")]
        [Space(6)]
        public IntegrationStage integrationStage;

        [Tooltip("The quality of the AO.")]
        [Space(10)]
        public Quality quality;

        [Tooltip("The deinterleaving factor.")]
        public Deinterleaving deinterleaving;

        [Tooltip("The resolution at which the AO is calculated.")]
        public Resolution resolution;

        [Tooltip("The type of noise to use.")]
        [Space(10)]
        public NoiseType noiseType;

        [Tooltip("The way the AO is displayed on screen.")]
        [Space(10)]
        public DisplayMode displayMode;

        [SerializeField]
        public static GeneralSettings defaultSettings
        {
            get
            {
                return new GeneralSettings
                {
                    integrationStage = IntegrationStage.BeforeImageEffectsOpaque,
                    quality = Quality.Highest,
                    deinterleaving = Deinterleaving.Disabled,
                    resolution = Resolution.Full,
                    noiseType = NoiseType.Dither,
                    displayMode = DisplayMode.AOOnly
                };
            }

        }
    }

    [Serializable]
    public struct AOSettings
    {
        [Tooltip("AO radius: this is the distance outside which occluders are ignored.")]
        [Space(6), Range(0, 35)]
        public float radius;

        [Tooltip("Maximum radius in pixels: this prevents the radius to grow too much with close-up " +
                  "object and impact on performances.")]
        [Range(64, 512)]
        public float maxRadiusPixels;

        [Tooltip("For low-tessellated geometry, occlusion variations tend to appear at creases and " +
                 "ridges, which betray the underlying tessellation. To remove these artifacts, we use " +
                 "an angle bias parameter which restricts the hemisphere.")]
        [Range(0, 0.5f)]
        public float bias;

        [Tooltip("This value allows to scale up the ambient occlusion values.")]
        [Range(0, 10)]
        public float intensity;

        [Tooltip("Enable/disable MultiBounce approximation.")]
        public bool useMultiBounce;

        [Tooltip("MultiBounce approximation influence.")]
        [Range(0, 1)]
        public float multiBounceInfluence;

        [Tooltip("The amount of AO offscreen samples are contributing.")]
        [Range(0, 1)]
        public float offscreenSamplesContribution;

        [Tooltip("The max distance to display AO.")]
        [Space(10)]
        public float maxDistance;

        [Tooltip("The distance before max distance at which AO start to decrease.")]
        public float distanceFalloff;

        [Tooltip("The type of per pixel normals to use.")]
        [Space(10)]
        public PerPixelNormals perPixelNormals;

        [Tooltip("This setting allow you to set the base color if the AO, the alpha channel value is unused.")]
        [Space(10)]
        public Color baseColor;

        [SerializeField]
        public static AOSettings defaultSettings
        {
            get
            {
                return new AOSettings
                {
                    radius = 25.0f,
                    maxRadiusPixels = 128f,
                    bias = 0.05f,
                    intensity = 1f,
                    useMultiBounce = false,
                    multiBounceInfluence = 1f,
                    offscreenSamplesContribution = 0f,
                    maxDistance = 1500f,
                    distanceFalloff = 0f,
                    perPixelNormals = PerPixelNormals.GBuffer,
                    baseColor = Color.black
                };
            }

        }

        public void set_radius(float r)
        {
            radius = r;
        }

        public float get_radius()
        {
            return radius;
        }

    }

    [Serializable]
    public struct ColorBleedingSettings
    {
        [Space(6)]
        public bool enabled;

        [Tooltip("This value allows to control the saturation of the color bleeding.")]
        [Space(10), Range(0, 4)]
        public float saturation;

        [Tooltip("This value allows to scale the contribution of the color bleeding samples.")]
        [Range(0, 32)]
        public float albedoMultiplier;

        [Tooltip("Use masking on emissive pixels")]
        [Range(0, 1)]
        public float brightnessMask;

        [Tooltip("Brightness level where masking starts/ends")]
        [HBAO_MinMaxSlider(0, 8)]
        public Vector2 brightnessMaskRange;

        [SerializeField]
        public static ColorBleedingSettings defaultSettings
        {
            get
            {
                return new ColorBleedingSettings
                {
                    enabled = false,
                    saturation = 1f,
                    albedoMultiplier = 4f,
                    brightnessMask = 1f,
                    brightnessMaskRange = new Vector2(0.8f, 1.2f)
                };
            }
        }
    }

    [Serializable]
    public struct BlurSettings
    {
        [Tooltip("The type of blur to use.")]
        [Space(6)]
        public Blur amount;

        [Tooltip("This parameter controls the depth-dependent weight of the bilateral filter, to " +
                 "avoid bleeding across edges. A zero sharpness is a pure Gaussian blur. Increasing " +
                 "the blur sharpness removes bleeding by using lower weights for samples with large " +
                 "depth delta from the current pixel.")]
        [Space(10), Range(0, 16)]
        public float sharpness;

        [Tooltip("Is the blur downsampled.")]
        public bool downsample;

        [SerializeField]
        public static BlurSettings defaultSettings
        {
            get
            {
                return new BlurSettings
                {
                    amount = Blur.Medium,
                    sharpness = 8f,
                    downsample = false
                };
            }
        }
    }

    [AttributeUsage(AttributeTargets.Field)]
    public class SettingsGroup : Attribute { }

    [SerializeField, SettingsGroup]
    private Presets m_Presets = Presets.defaultPresets;
    public Presets presets
    {
        get { return m_Presets; }
        set { m_Presets = value; }
    }

    [SerializeField, SettingsGroup]
    private GeneralSettings m_GeneralSettings = GeneralSettings.defaultSettings;
    public GeneralSettings generalSettings
    {
        get { return m_GeneralSettings; }
        set { m_GeneralSettings = value; }
    }

    [SerializeField, SettingsGroup]
    private AOSettings m_AOSettings = AOSettings.defaultSettings;
    public AOSettings aoSettings
    {
        get { return m_AOSettings; }
        set { m_AOSettings = value; }
    }

    [SerializeField, SettingsGroup]
    private ColorBleedingSettings m_ColorBleedingSettings = ColorBleedingSettings.defaultSettings;
    public ColorBleedingSettings colorBleedingSettings
    {
        get { return m_ColorBleedingSettings; }
        set { m_ColorBleedingSettings = value; }
    }

    [SerializeField, SettingsGroup]
    private BlurSettings m_BlurSettings = BlurSettings.defaultSettings;
    public BlurSettings blurSettings
    {
        get { return m_BlurSettings; }
        set { m_BlurSettings = value; }
    }

    private static class MersenneTwister
    {
        // Mersenne-Twister random numbers in [0,1).
        public static float[] Numbers = new float[] {
            0.463937f,0.340042f,0.223035f,0.468465f,0.322224f,0.979269f,0.031798f,0.973392f,0.778313f,0.456168f,0.258593f,0.330083f,0.387332f,0.380117f,0.179842f,0.910755f,
            0.511623f,0.092933f,0.180794f,0.620153f,0.101348f,0.556342f,0.642479f,0.442008f,0.215115f,0.475218f,0.157357f,0.568868f,0.501241f,0.629229f,0.699218f,0.707733f
        };
    }

    protected static class Pass
    {
        public const int AO_LowestQuality = 0;
        public const int AO_LowQuality = 1;
        public const int AO_MediumQuality = 2;
        public const int AO_HighQuality = 3;
        public const int AO_HighestQuality = 4;
        public const int AO_Deinterleaved_LowestQuality = 5;
        public const int AO_Deinterleaved_LowQuality = 6;
        public const int AO_Deinterleaved_MediumQuality = 7;
        public const int AO_Deinterleaved_HighQuality = 8;
        public const int AO_Deinterleaved_HighestQuality = 9;

        public const int Depth_Deinterleaving_2x2 = 10;
        public const int Depth_Deinterleaving_4x4 = 11;
        public const int Normals_Deinterleaving_2x2 = 12;
        public const int Normals_Deinterleaving_4x4 = 13;

        public const int Atlas = 14;

        public const int Reinterleaving_2x2 = 15;
        public const int Reinterleaving_4x4 = 16;

        public const int Blur_X_Narrow = 17;
        public const int Blur_X_Medium = 18;
        public const int Blur_X_Wide = 19;
        public const int Blur_X_ExtraWide = 20;
        public const int Blur_Y_Narrow = 21;
        public const int Blur_Y_Medium = 22;
        public const int Blur_Y_Wide = 23;
        public const int Blur_Y_ExtraWide = 24;

        public const int Composite = 25;
        public const int Composite_MultiBounce = 26;
        public const int Debug_AO_Only = 27;
        public const int Debug_AO_Only_MultiBounce = 28;
        public const int Debug_ColorBleeding_Only = 29;
        public const int Debug_Split_WithoutAO_WithAO = 30;
        public const int Debug_Split_WithoutAO_WithAO_MultiBounce = 31;
        public const int Debug_Split_WithAO_AOOnly = 32;
        public const int Debug_Split_WithAO_AOOnly_MultiBounce = 33;
        public const int Debug_Split_WithoutAO_AOOnly = 34;
        public const int Debug_Split_WithoutAO_AOOnly_MultiBounce = 35;

        public const int Combine_Deffered = 36;
        public const int Combine_Deffered_Multiplicative = 37;
        public const int Combine_Integrated = 38;
        public const int Combine_Integrated_MultiBounce = 39;
        public const int Combine_Integrated_Multiplicative = 40;
        public const int Combine_Integrated_Multiplicative_MultiBounce = 41;
        public const int Combine_ColorBleeding = 42;
        public const int Debug_Split_Additive = 43;
        public const int Debug_Split_Additive_MultiBounce = 44;
        public const int Debug_Split_Multiplicative = 45;
        public const int Debug_Split_Multiplicative_MultiBounce = 46;
    }

    protected class RenderTarget
    {
        public bool orthographic;
        public RenderingPath renderingPath;
        public bool hdr;
        public int width;
        public int height;
        public int fullWidth;
        public int fullHeight;
        public int layerWidth;
        public int layerHeight;
        public int downsamplingFactor;
        public int deinterleavingFactor;
        public int blurDownsamplingFactor;
    }

    protected static class ShaderProperties
    {
        public static int mainTex;
        public static int hbaoTex;
        public static int noiseTex;


        public static int mydepthTex;
        public static int mynormalTex;


        public static int rt0Tex;
        public static int rt3Tex;
        public static int depthTex;
        public static int normalsTex;
        public static int[] mrtDepthTex;
        public static int[] mrtNrmTex;
        public static int[] mrtHBAOTex;
        public static int[] deinterleavingOffset;
        public static int layerOffset;
        public static int jitter;
        public static int uvToView;
        public static int worldToCameraMatrix;
        public static int fullResTexelSize;
        public static int layerResTexelSize;
        public static int targetScale;
        public static int noiseTexSize;
        public static int radius;
        public static int maxRadiusPixels;
        public static int negInvRadius2;
        public static int angleBias;
        public static int aoMultiplier;
        public static int intensity;
        public static int multiBounceInfluence;
        public static int offscreenSamplesContrib;
        public static int maxDistance;
        public static int distanceFalloff;
        public static int baseColor;
        public static int colorBleedSaturation;
        public static int albedoMultiplier;
        public static int colorBleedBrightnessMask;
        public static int colorBleedBrightnessMaskRange;
        public static int blurSharpness;


        static ShaderProperties()
        {
            mainTex = Shader.PropertyToID("_MainTex");
            hbaoTex = Shader.PropertyToID("_HBAOTex");

            mydepthTex = Shader.PropertyToID("_MyDepthTex");
            mynormalTex = Shader.PropertyToID("_MyNormalTex");

            noiseTex = Shader.PropertyToID("_NoiseTex");
            rt0Tex = Shader.PropertyToID("_rt0Tex");
            rt3Tex = Shader.PropertyToID("_rt3Tex");
            depthTex = Shader.PropertyToID("_DepthTex");
            normalsTex = Shader.PropertyToID("_NormalsTex");  //-------------------NOAMAL---------------------//
            mrtDepthTex = new int[4 * NUM_MRTS];
            mrtNrmTex = new int[4 * NUM_MRTS];
            mrtHBAOTex = new int[4 * NUM_MRTS];
            for (int i = 0; i < 4 * NUM_MRTS; i++)
            {
                mrtDepthTex[i] = Shader.PropertyToID("_DepthLayerTex" + i);
                mrtNrmTex[i] = Shader.PropertyToID("_NormalLayerTex" + i);
                mrtHBAOTex[i] = Shader.PropertyToID("_HBAOLayerTex" + i);
            }

            deinterleavingOffset = new int[] {
                Shader.PropertyToID("_Deinterleaving_Offset00"),
                Shader.PropertyToID("_Deinterleaving_Offset10"),
                Shader.PropertyToID("_Deinterleaving_Offset01"),
                Shader.PropertyToID("_Deinterleaving_Offset11")
            };

            layerOffset = Shader.PropertyToID("_LayerOffset");
            jitter = Shader.PropertyToID("_Jitter");
            uvToView = Shader.PropertyToID("_UVToView");
            worldToCameraMatrix = Shader.PropertyToID("_WorldToCameraMatrix");
            fullResTexelSize = Shader.PropertyToID("_FullRes_TexelSize");
            layerResTexelSize = Shader.PropertyToID("_LayerRes_TexelSize");
            targetScale = Shader.PropertyToID("_TargetScale");
            noiseTexSize = Shader.PropertyToID("_NoiseTexSize");
            radius = Shader.PropertyToID("_Radius");
            maxRadiusPixels = Shader.PropertyToID("_MaxRadiusPixels");
            negInvRadius2 = Shader.PropertyToID("_NegInvRadius2");
            angleBias = Shader.PropertyToID("_AngleBias");
            aoMultiplier = Shader.PropertyToID("_AOmultiplier");
            intensity = Shader.PropertyToID("_Intensity");
            multiBounceInfluence = Shader.PropertyToID("_MultiBounceInfluence");
            offscreenSamplesContrib = Shader.PropertyToID("_OffscreenSamplesContrib");
            maxDistance = Shader.PropertyToID("_MaxDistance");
            distanceFalloff = Shader.PropertyToID("_DistanceFalloff");
            baseColor = Shader.PropertyToID("_BaseColor");
            colorBleedSaturation = Shader.PropertyToID("_ColorBleedSaturation");
            albedoMultiplier = Shader.PropertyToID("_AlbedoMultiplier");
            colorBleedBrightnessMask = Shader.PropertyToID("_ColorBleedBrightnessMask");
            colorBleedBrightnessMaskRange = Shader.PropertyToID("_ColorBleedBrightnessMaskRange");
            blurSharpness = Shader.PropertyToID("_BlurSharpness");
        }
    }

    protected Material _hbaoMaterial;
    protected Camera _hbaoCamera;
    protected RenderTarget _renderTarget;
    protected const int NUM_MRTS = 4;
    protected const int NUM_SANPLES = 100;

    protected Vector4[] _jitter = new Vector4[4 * NUM_MRTS];

    private Quality _quality;
    private NoiseType _noiseType;
    private string[] _hbaoShaderKeywords = new string[4];
    private int[] _numSampleDirections = new int[] { 3, 4, 6, 8, 8 }; // LOWEST, LOW, MEDIUM, HIGH, HIGHEST (highest uses more steps)

    protected virtual void OnEnable()
    {
        if (!SystemInfo.supportsImageEffects || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.Depth))
        {
            Debug.LogWarning("HBAO shader is not supported on this platform.");
            this.enabled = false;
            return;
        }

        if (hbaoShader != null && !hbaoShader.isSupported)
        {
            Debug.LogWarning("HBAO shader is not supported on this platform.");
            this.enabled = false;
            return;
        }

        if (hbaoShader == null)
        {
            return;
        }

        CreateMaterial();

        _hbaoCamera.depthTextureMode |= DepthTextureMode.Depth;
        if (aoSettings.perPixelNormals == PerPixelNormals.Camera)
            _hbaoCamera.depthTextureMode |= DepthTextureMode.DepthNormals;
#if UNITY_5_6_OR_NEWER
        _hbaoCamera.forceIntoRenderTexture = true;
#endif
    }

    protected virtual void OnDisable()
    {
        if (_hbaoMaterial != null)
            DestroyImmediate(_hbaoMaterial);
        if (noiseTex != null)
            DestroyImmediate(noiseTex);


        if (mynormalTex != null)
            DestroyImmediate(mynormalTex);
        if (mydepthTex != null)
            DestroyImmediate(mydepthTex);


        if (quadMesh != null)
            DestroyImmediate(quadMesh);
    }

    private void CreateMaterial()
    {
        if (_hbaoMaterial == null)
        {
            _hbaoMaterial = new Material(hbaoShader);
            _hbaoMaterial.hideFlags = HideFlags.HideAndDontSave;

            _hbaoCamera = GetComponent<Camera>();
        }

        if (quadMesh != null)
            DestroyImmediate(quadMesh);

        quadMesh = new Mesh();
        quadMesh.vertices = new Vector3[] {
            new Vector3(-0.5f, -0.5f, 0),
            new Vector3( 0.5f,  0.5f, 0),
            new Vector3( 0.5f, -0.5f, 0),
            new Vector3(-0.5f,  0.5f, 0)
        };
        quadMesh.uv = new Vector2[] {
            new Vector2(0, 0),
            new Vector2(1, 1),
            new Vector2(1, 0),
            new Vector2(0, 1)
        };
        quadMesh.triangles = new int[] { 0, 1, 2, 1, 0, 3 };

        _renderTarget = new RenderTarget();
    }

    //public static Texture2D GetTexrture2DFromPath(string imgPath)
    //{
    //    //读取文件
    //    FileStream fs = new FileStream(imgPath, FileMode.Open, FileAccess.Read);
    //    int byteLength = (int)fs.Length;
    //    byte[] imgBytes = new byte[byteLength];
    //    fs.Read(imgBytes, 0, byteLength);
    //    fs.Close();
    //    fs.Dispose();
    //    //转化为Texture2D
    //    Image img = Image.FromStream(new MemoryStream(imgBytes));
    //    Texture2D t2d = new Texture2D(img.Width, img.Height);
    //    img.Dispose();
    //    t2d.LoadImage(imgBytes);
    //    t2d.Apply();
    //    return t2d;
    //}


    protected void UpdateShaderProperties(int num)
    {
        _renderTarget.orthographic = _hbaoCamera.orthographic;
        _renderTarget.renderingPath = _hbaoCamera.actualRenderingPath;
#if UNITY_5_6_OR_NEWER
        _renderTarget.hdr = _hbaoCamera.allowHDR;
#else
        _renderTarget.hdr = _hbaoCamera.hdr;
#endif
#if UNITY_2017_2_OR_NEWER
        if (UnityEngine.XR.XRSettings.enabled)
        {
            _renderTarget.width = UnityEngine.XR.XRSettings.eyeTextureDesc.width;
            _renderTarget.height = UnityEngine.XR.XRSettings.eyeTextureDesc.height;
        }
        else
        {
            _renderTarget.width = _hbaoCamera.pixelWidth;
            _renderTarget.height = _hbaoCamera.pixelHeight;
        }
#else
        _renderTarget.width = _hbaoCamera.pixelWidth;
        _renderTarget.height = _hbaoCamera.pixelHeight;
#endif
        _renderTarget.downsamplingFactor = generalSettings.resolution == Resolution.Full ? 1 : generalSettings.resolution == Resolution.Half ? 2 : 4;
        _renderTarget.deinterleavingFactor = GetDeinterleavingFactor();
        _renderTarget.blurDownsamplingFactor = blurSettings.downsample ? 2 : 1;

        //float tanHalfFovY = Mathf.Tan(0.5f * _hbaoCamera.fieldOfView * Mathf.Deg2Rad);
        //float invFocalLenX = 1.0f / (1.0f / tanHalfFovY * (_renderTarget.height / (float)_renderTarget.width));
        //float invFocalLenY = 1.0f / (1.0f / tanHalfFovY);
        //_hbaoMaterial.SetVector(ShaderProperties.uvToView, new Vector4(2.0f * invFocalLenX, -2.0f * invFocalLenY, -1.0f * invFocalLenX, 1.0f * invFocalLenY));

        float tanHalfFovY = 0.4663077f;
        _hbaoMaterial.SetVector(ShaderProperties.uvToView, new Vector4(2.0f * tanHalfFovY, -2.0f * tanHalfFovY, -1.0f * tanHalfFovY, 1.0f * tanHalfFovY));


        _hbaoMaterial.SetMatrix(ShaderProperties.worldToCameraMatrix, _hbaoCamera.worldToCameraMatrix);

        if (generalSettings.deinterleaving != Deinterleaving.Disabled)
        {
            _renderTarget.fullWidth = _renderTarget.width + (_renderTarget.width % _renderTarget.deinterleavingFactor == 0 ? 0 : _renderTarget.deinterleavingFactor - (_renderTarget.width % _renderTarget.deinterleavingFactor));
            _renderTarget.fullHeight = _renderTarget.height + (_renderTarget.height % _renderTarget.deinterleavingFactor == 0 ? 0 : _renderTarget.deinterleavingFactor - (_renderTarget.height % _renderTarget.deinterleavingFactor));
            _renderTarget.layerWidth = _renderTarget.fullWidth / _renderTarget.deinterleavingFactor;
            _renderTarget.layerHeight = _renderTarget.fullHeight / _renderTarget.deinterleavingFactor;

            _hbaoMaterial.SetVector(ShaderProperties.fullResTexelSize, new Vector4(1.0f / _renderTarget.fullWidth, 1.0f / _renderTarget.fullHeight, _renderTarget.fullWidth, _renderTarget.fullHeight));
            _hbaoMaterial.SetVector(ShaderProperties.layerResTexelSize, new Vector4(1.0f / _renderTarget.layerWidth, 1.0f / _renderTarget.layerHeight, _renderTarget.layerWidth, _renderTarget.layerHeight));
            _hbaoMaterial.SetVector(ShaderProperties.targetScale, new Vector4(_renderTarget.fullWidth / (float)_renderTarget.width, _renderTarget.fullHeight / (float)_renderTarget.height, 1.0f / (_renderTarget.fullWidth / (float)_renderTarget.width), 1.0f / (_renderTarget.fullHeight / (float)_renderTarget.height)));
        }
        else
        {
            _renderTarget.fullWidth = _renderTarget.width;
            _renderTarget.fullHeight = _renderTarget.height;
            if (generalSettings.resolution == Resolution.Half && aoSettings.perPixelNormals == PerPixelNormals.Reconstruct)
                _hbaoMaterial.SetVector(ShaderProperties.targetScale, new Vector4((_renderTarget.width + 0.5f) / _renderTarget.width, (_renderTarget.height + 0.5f) / _renderTarget.height, 1f, 1f));
            else
                _hbaoMaterial.SetVector(ShaderProperties.targetScale, new Vector4(1f, 1f, 1f, 1f));
        }

        if (noiseTex == null || _quality != generalSettings.quality || _noiseType != generalSettings.noiseType)
        {
            if (noiseTex != null)
                DestroyImmediate(noiseTex);

            float noiseTexSize = generalSettings.noiseType == NoiseType.Dither ? 4 : 64;
            CreateRandomTexture((int)noiseTexSize);
        }

        //if (mydepthnormalTex == null)
        //{
        //    mydepthnormalTex = new Texture2D(512, 512, TextureFormat.RGBAFloat, false, true);
        //    byte[] fileData;
        //    string filePath = "D:\\Projects\\Unity\\test_03\\screenshot\\2-C-Depth.exr";
        //    fileData = File.ReadAllBytes(filePath);
        //    mydepthnormalTex.LoadImage(fileData);
        //    mydepthnormalTex.Apply();
        //}


        //mydepthTex = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        //byte[] fileData;
        //string filePath = "D:\\Projects\\Unity\\test_03\\screenshot\\depth-3.png";
        //fileData = File.ReadAllBytes(filePath);

        //mydepthTex.LoadImage(fileData);
        //mydepthTex.Apply();

        //mynormalTex = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        //byte[] fileData_normal;
        ////int num = 2;
        //string filePath_normal = "D:\\Projects\\Unity\\test_03\\screenshot\\normal-3.png";
        ////string filePath_normal = "D:\\Projects\\Unity\\test_03\\screenshot\\normal-" + num.ToString() + ".png";
        //fileData_normal = File.ReadAllBytes(filePath_normal);

        //mynormalTex.LoadImage(fileData_normal);
        //mynormalTex.Apply();



        _quality = generalSettings.quality;
        _noiseType = generalSettings.noiseType;

        _hbaoMaterial.SetTexture(ShaderProperties.noiseTex, noiseTex);

        //_hbaoMaterial.SetTexture(ShaderProperties.mydepthTex, mydepthTex);
        //_hbaoMaterial.SetTexture(ShaderProperties.mynormalTex, mynormalTex);

        _hbaoMaterial.SetFloat(ShaderProperties.noiseTexSize, _noiseType == NoiseType.Dither ? 4 : 64);
        _hbaoMaterial.SetFloat(ShaderProperties.radius, aoSettings.radius * 0.5f * (_renderTarget.height / (tanHalfFovY * 2.0f)) / _renderTarget.deinterleavingFactor);
        _hbaoMaterial.SetFloat(ShaderProperties.maxRadiusPixels, aoSettings.maxRadiusPixels / _renderTarget.deinterleavingFactor);
        _hbaoMaterial.SetFloat(ShaderProperties.negInvRadius2, -1.0f / (aoSettings.radius * aoSettings.radius));
        _hbaoMaterial.SetFloat(ShaderProperties.angleBias, aoSettings.bias);
        _hbaoMaterial.SetFloat(ShaderProperties.aoMultiplier, 2.0f * (1.0f / (1.0f - aoSettings.bias)));
        _hbaoMaterial.SetFloat(ShaderProperties.intensity, aoSettings.intensity);
        _hbaoMaterial.SetFloat(ShaderProperties.multiBounceInfluence, aoSettings.multiBounceInfluence);
        _hbaoMaterial.SetFloat(ShaderProperties.offscreenSamplesContrib, aoSettings.offscreenSamplesContribution);
        _hbaoMaterial.SetFloat(ShaderProperties.maxDistance, aoSettings.maxDistance);
        _hbaoMaterial.SetFloat(ShaderProperties.distanceFalloff, aoSettings.distanceFalloff);
        _hbaoMaterial.SetColor(ShaderProperties.baseColor, aoSettings.baseColor);
        _hbaoMaterial.SetFloat(ShaderProperties.colorBleedSaturation, colorBleedingSettings.saturation);
        _hbaoMaterial.SetFloat(ShaderProperties.albedoMultiplier, colorBleedingSettings.albedoMultiplier);
        _hbaoMaterial.SetFloat(ShaderProperties.colorBleedBrightnessMask, colorBleedingSettings.brightnessMask);
        _hbaoMaterial.SetVector(ShaderProperties.colorBleedBrightnessMaskRange, colorBleedingSettings.brightnessMaskRange);
        _hbaoMaterial.SetFloat(ShaderProperties.blurSharpness, blurSettings.sharpness);
    }

    protected void UpdateShaderKeywords()
    {
        _hbaoShaderKeywords[0] = colorBleedingSettings.enabled ? "COLOR_BLEEDING_ON" : "__";

        if (_renderTarget.orthographic)
            _hbaoShaderKeywords[1] = "ORTHOGRAPHIC_PROJECTION_ON";
        else
            _hbaoShaderKeywords[1] = IsDeferredShading() ? "DEFERRED_SHADING_ON" : "__";

        _hbaoShaderKeywords[2] = aoSettings.perPixelNormals == PerPixelNormals.Camera ? "NORMALS_CAMERA" : aoSettings.perPixelNormals == PerPixelNormals.Reconstruct ? "NORMALS_RECONSTRUCT" : "__";

        _hbaoShaderKeywords[3] = aoSettings.offscreenSamplesContribution > 0 ? "OFFSCREEN_SAMPLES_CONTRIB" : "__";

        _hbaoMaterial.shaderKeywords = _hbaoShaderKeywords;
    }

    protected virtual void CheckParameters()
    {
        if (!IsDeferredShading() && aoSettings.perPixelNormals == PerPixelNormals.GBuffer)
            m_AOSettings.perPixelNormals = PerPixelNormals.Camera;

        if (generalSettings.deinterleaving != Deinterleaving.Disabled && SystemInfo.supportedRenderTargetCount < 4)
            m_GeneralSettings.deinterleaving = Deinterleaving.Disabled;
    }

    protected bool IsDeferredShading()
    {
        return _hbaoCamera.actualRenderingPath == RenderingPath.DeferredShading;
    }

    protected bool IsDeferredShadingOrLighting()
    {
        return _hbaoCamera.actualRenderingPath == RenderingPath.DeferredShading || _hbaoCamera.actualRenderingPath == RenderingPath.DeferredLighting;
    }

    protected int GetDeinterleavingFactor()
    {
        switch (generalSettings.deinterleaving)
        {
            case Deinterleaving._2x:
                return 2;
            case Deinterleaving._4x:
                return 4;
            case Deinterleaving.Disabled:
            default:
                return 1;
        }
    }

    protected int GetAoPass()
    {
        switch (generalSettings.quality)
        {
            case Quality.Lowest:
                return Pass.AO_LowestQuality;
            case Quality.Low:
                return Pass.AO_LowQuality;
            case Quality.Medium:
                return Pass.AO_MediumQuality;
            case Quality.High:
                return Pass.AO_HighQuality;
            case Quality.Highest:
                return Pass.AO_HighestQuality;
            default:
                return Pass.AO_MediumQuality;
        }
    }

    protected int GetAoDeinterleavedPass()
    {
        switch (generalSettings.quality)
        {
            case Quality.Lowest:
                return Pass.AO_Deinterleaved_LowestQuality;
            case Quality.Low:
                return Pass.AO_Deinterleaved_LowQuality;
            case Quality.Medium:
                return Pass.AO_Deinterleaved_MediumQuality;
            case Quality.High:
                return Pass.AO_Deinterleaved_HighQuality;
            case Quality.Highest:
                return Pass.AO_Deinterleaved_HighestQuality;
            default:
                return Pass.AO_Deinterleaved_MediumQuality;
        }
    }

    protected int GetBlurXPass()
    {
        switch (blurSettings.amount)
        {
            case Blur.Narrow:
                return Pass.Blur_X_Narrow;
            case Blur.Medium:
                return Pass.Blur_X_Medium;
            case Blur.Wide:
                return Pass.Blur_X_Wide;
            case Blur.ExtraWide:
                return Pass.Blur_X_ExtraWide;
            default:
                return Pass.Blur_X_Medium;
        }
    }

    protected int GetBlurYPass()
    {
        switch (blurSettings.amount)
        {
            case Blur.Narrow:
                return Pass.Blur_Y_Narrow;
            case Blur.Medium:
                return Pass.Blur_Y_Medium;
            case Blur.Wide:
                return Pass.Blur_Y_Wide;
            case Blur.ExtraWide:
                return Pass.Blur_Y_ExtraWide;
            default:
                return Pass.Blur_Y_Medium;
        }
    }

    protected int GetFinalPass()
    {
        switch (generalSettings.displayMode)
        {
            case DisplayMode.Normal:
                return aoSettings.useMultiBounce ? Pass.Composite_MultiBounce : Pass.Composite;
            case DisplayMode.AOOnly:
                return aoSettings.useMultiBounce ? Pass.Debug_AO_Only_MultiBounce : Pass.Debug_AO_Only;
            case DisplayMode.ColorBleedingOnly:
                return Pass.Debug_ColorBleeding_Only;
            case DisplayMode.SplitWithoutAOAndWithAO:
                return aoSettings.useMultiBounce ? Pass.Debug_Split_WithoutAO_WithAO_MultiBounce : Pass.Debug_Split_WithoutAO_WithAO;
            case DisplayMode.SplitWithAOAndAOOnly:
                return aoSettings.useMultiBounce ? Pass.Debug_Split_WithAO_AOOnly_MultiBounce : Pass.Debug_Split_WithAO_AOOnly;
            case DisplayMode.SplitWithoutAOAndAOOnly:
                return aoSettings.useMultiBounce ? Pass.Debug_Split_WithoutAO_AOOnly_MultiBounce : Pass.Debug_Split_WithoutAO_AOOnly;
            default:
                return Pass.Composite;
        }
    }

    private void CreateRandomTexture(int size)
    {
        noiseTex = new Texture2D(size, size, TextureFormat.RGB24, false, true);
        noiseTex.filterMode = FilterMode.Point;
        noiseTex.wrapMode = TextureWrapMode.Repeat;
        int z = 0;
        for (int x = 0; x < size; ++x)
        {
            for (int y = 0; y < size; ++y)
            {
                float r1 = generalSettings.noiseType == NoiseType.Dither ? MersenneTwister.Numbers[z++] : UnityEngine.Random.Range(0.0f, 1.0f);
                float r2 = generalSettings.noiseType == NoiseType.Dither ? MersenneTwister.Numbers[z++] : UnityEngine.Random.Range(0.0f, 1.0f);
                float angle = 2.0f * Mathf.PI * r1 / _numSampleDirections[GetAoPass()];
                Color color = new Color(Mathf.Cos(angle), Mathf.Sin(angle), r2);
                noiseTex.SetPixel(x, y, color);
            }
        }
        noiseTex.Apply();

        for (int i = 0, j = 0; i < _jitter.Length; ++i)
        {
            float r1 = MersenneTwister.Numbers[j++];
            float r2 = MersenneTwister.Numbers[j++];
            float angle = 2.0f * Mathf.PI * r1 / _numSampleDirections[GetAoPass()];
            _jitter[i] = new Vector4(Mathf.Cos(angle), Mathf.Sin(angle), r2, 0);
        }
    }

    public void ApplyPreset(Preset preset)
    {
        if (preset == Preset.Custom)
        {
            m_Presets.preset = preset;
            return;
        }

        DisplayMode displayMode = generalSettings.displayMode;

        m_GeneralSettings = GeneralSettings.defaultSettings;
        m_AOSettings = AOSettings.defaultSettings;
        m_ColorBleedingSettings = ColorBleedingSettings.defaultSettings;
        m_BlurSettings = BlurSettings.defaultSettings;

        m_GeneralSettings.displayMode = displayMode;

        switch (preset)
        {
            case Preset.FastestPerformance:
                m_GeneralSettings.quality = Quality.Lowest;
                m_AOSettings.radius = 0.5f;
                m_AOSettings.maxRadiusPixels = 64.0f;
                m_BlurSettings.amount = Blur.ExtraWide;
                m_BlurSettings.downsample = true;
                break;
            case Preset.FastPerformance:
                m_GeneralSettings.quality = Quality.Low;
                m_AOSettings.radius = 0.5f;
                m_AOSettings.maxRadiusPixels = 64.0f;
                m_BlurSettings.amount = Blur.Wide;
                m_BlurSettings.downsample = true;
                break;
            case Preset.HighQuality:
                m_GeneralSettings.quality = Quality.High;
                m_AOSettings.radius = 1.0f;
                break;
            case Preset.HighestQuality:
                m_GeneralSettings.quality = Quality.Highest;
                m_AOSettings.radius = 1.2f;
                m_AOSettings.maxRadiusPixels = 256.0f;
                m_BlurSettings.amount = Blur.Narrow;
                break;
            case Preset.Normal:
            default:
                break;
        }

        m_Presets.preset = preset;
    }
}







