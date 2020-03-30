// Copyright (c) 2016-2018 Jakub Boksansky - All Rights Reserved
// Volumetric Ambient Occlusion Unity Plugin 2.0

using UnityEngine;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.AnimatedValues;
#endif
using System;
using System.Collections.Generic;
using System.Reflection;
using System.IO;

namespace Wilberforce.VAO
{

    [ExecuteInEditMode]
    [RequireComponent(typeof(Camera))]
    [HelpURL("https://projectwilberforce.github.io/vaomanual/")]
    public class VAOEffectCommandBuffer : MonoBehaviour
    {

        #region VAO Parameters

        /// <summary>
        /// Radius of AO calculation
        /// </summary>
        public float Radius = 0.5f;

        /// <summary>
        /// Intensity of AO effect
        /// </summary>
        public float Power = 1.0f;

        /// <summary>
        /// Intensity of far occlusion - higher makes AO more pronounced further from occluding object
        /// </summary>
        public float Presence = 0.1f;

        /// <summary>
        /// Thickness of occlusion behind what is seen by camera - controls halo around objects and "depth" of occlusion behind objects
        /// </summary>
        public float Thickness = 0.25f;

        /// <summary>
        /// Sets balance between occlusion on screen borders and flickering (popping)
        /// </summary>
        public float BordersIntensity = 0.3f;

        /// <summary>
        /// Amount of fine detail captured by VAO algorithm - more detail affects performance
        /// </summary>
        public float DetailAmountVAO = 0.0f;

        /// <summary>
        /// Number of samples used to calculate "detailed" shadows controlled by DetailAmount vatriable
        /// </summary>
        public DetailQualityType DetailQuality = DetailQualityType.Medium;

        /// <summary>
        /// Amount of fine detail captured by Raycast AO algorithm - more detail affects performance
        /// </summary>
        public float DetailAmountRaycast = 0.0f;

        /// <summary>
        /// Number of samples to use - 2,4,8,16,32 or 64.
        /// In case of adaptive sampling, this is maximum number of samples that will be used.
        /// </summary>
        public int Quality = 64;

        /// <summary>
        /// Bias adjustment for classic SSAO and raycast SSAO algorithms
        /// </summary>
        public float SSAOBias = 0.005f;

        #region Near/Far Occlusion Limits

        /// <summary>
        /// When enabled, radius is limited to uper bound in screen space units to save performance
        /// </summary>
        public bool MaxRadiusEnabled = true;

        /// <summary>
        /// Maximal radius limit wrt. to screen size.
        /// Should be set so that large AO radius for objects close to camera will not cause performance drop
        /// </summary>
        public float MaxRadius = 0.5f;

        /// <summary>
        /// Ambient Occlusion algorithm to use
        /// </summary>
        public AlgorithmType Algorithm = AlgorithmType.StandardVAO;

        /// <summary>
        /// Mode for distance suppression of AO. Relative mode lets user set distance in units relative to far plane.
        /// Absolute mode uses world space units.
        /// </summary>
        public DistanceFalloffModeType DistanceFalloffMode = DistanceFalloffModeType.Off;

        /// <summary>
        /// Distance from which will be AO reduced (in world space units)
        /// </summary>
        public float DistanceFalloffStartAbsolute = 100.0f;

        /// <summary>
        /// Distance from which will be AO reduced (relative to far plane)
        /// </summary>
        public float DistanceFalloffStartRelative = 0.1f;

        /// <summary>
        /// Distance from "falloff start" in which is AO reduced  to zero (in world space units)
        /// </summary>
        public float DistanceFalloffSpeedAbsolute = 30.0f;

        /// <summary>
        /// Distance from "falloff start" in which is AO reduced  to zero (relative to far plane)
        /// </summary>
        public float DistanceFalloffSpeedRelative = 0.1f;

        #endregion

        #region Performance Settings

        /// <summary>
        /// Adaptive sampling setting - off/auto/manual.
        /// Automatic decides nubmer of samples automatically, manual mode takes user parameter
        /// AdaptiveQualityCoefficient into account.
        /// </summary>
        public AdaptiveSamplingType AdaptiveType = AdaptiveSamplingType.EnabledAutomatic;

        /// <summary>
        /// Moves threshold where number of samples starts to decrease due to distance from camera
        /// </summary>
        public float AdaptiveQualityCoefficient = 1.0f;

        /// <summary>
        /// Culling prepass setting - off/greeddy/careful.
        /// Greedy will not calculate AO in areas where no occlusion is estimated.
        /// Careful will use only 4 samples in such areas to prevent loss of detail.
        /// </summary>
        public CullingPrepassModeType CullingPrepassMode = CullingPrepassModeType.Careful;

        /// <summary>
        /// AO resolution downsampling - possible values are: 1 (original resolution), 2 or 4.
        /// </summary>
        public int Downsampling = 1;

        /// <summary>
        /// Setting of hierarchical buffer - off/on/auto.
        /// Auto will turn hierarchical buffer on when radius is large enough based on HierarchicalBufferPixelsPerLevel variable
        /// </summary>
        public HierarchicalBufferStateType HierarchicalBufferState = HierarchicalBufferStateType.Auto;

        #endregion

        #region Rendering Pipeline

        /// <summary>
        /// Enable to use command buffer pipeline instead of image effect implementation
        /// </summary>
        public bool CommandBufferEnabled = true;

        /// <summary>
        /// Enable to use GBuffer normals and depth (can provide higher precision, but available only in deferred rendering path)
        /// </summary>
        public bool UseGBuffer = true;

        /// <summary>
        /// Use higher precision dedicated depth buffer. Can solve problems with some see-thorugh materials in deferred rendering path.
        /// </summary>
        public bool UsePreciseDepthBuffer = true;

        /// <summary>
        /// Enables temporal filtering (taking occlusion values of previous frames into account) for higher quality and performance.
        /// </summary>
        public bool EnableTemporalFiltering = true;

        /// <summary>
        /// Rendering stage in which will be VAO effect applied (available only in deferred path)
        /// </summary>
        public VAOCameraEventType VaoCameraEvent = VAOCameraEventType.AfterLighting;

        /// <summary>
        /// Where to take far plane distance from. Either from camera property or from Unity's built-in shader variable.
        /// </summary>
        public FarPlaneSourceType FarPlaneSource = FarPlaneSourceType.Camera;

        #endregion

        #region Luminance Sensitivity

        /// <summary>
        /// Enable luminance sensitivity - suppresion of occlusion based on luminance of shaded surface.
        /// </summary>
        public bool IsLumaSensitive = false;

        /// <summary>
        /// Luminance calculation formula - weighted RGB components (luma) or value component of HSV model.
        /// </summary>
        public LuminanceModeType LuminanceMode = LuminanceModeType.Luma;

        /// <summary>
        /// Threshold of luminance where suppresion is applied
        /// </summary>
        public float LumaThreshold = 0.7f;

        /// <summary>
        /// Controls width of gradual suppresion by luminance
        /// </summary>
        public float LumaKneeWidth = 0.3f;

        /// <summary>
        /// Controls shape of luminance suppresion curve
        /// </summary>
        public float LumaKneeLinearity = 3.0f;

        #endregion

        /// <summary>
        /// Effect Mode - simple (black occlusion), color tint, or colorbleeding
        /// </summary>
        public EffectMode Mode = EffectMode.ColorTint;

        /// <summary>
        /// Color tint applied to occlusion in ColorTint mode
        /// </summary>
        public Color ColorTint = Color.black;

        #region Color Bleeding Settings

        /// <summary>
        /// Intensity of color bleeding
        /// </summary>
        public float ColorBleedPower = 5.0f;

        /// <summary>
        /// Limits saturation of colorbleeding
        /// </summary>
        public float ColorBleedPresence = 1.0f;

        /// <summary>
        /// Format of texture used to store screen image on GPU
        /// </summary>
        public ScreenTextureFormat IntermediateScreenTextureFormat = ScreenTextureFormat.Auto;

        /// <summary>
        /// Enables suppresion of color bleeding from surfaces with same color (hue and saturation filter)
        /// </summary>
        public bool ColorbleedHueSuppresionEnabled = false;

        /// <summary>
        /// Tolerance range for hue filter
        /// </summary>
        public float ColorBleedHueSuppresionThreshold = 7.0f;

        /// <summary>
        /// Hue filter softness
        /// </summary>
        public float ColorBleedHueSuppresionWidth = 2.0f;

        /// <summary>
        /// Tolerance range for saturation filter
        /// </summary>
        public float ColorBleedHueSuppresionSaturationThreshold = 0.5f;

        /// <summary>
        /// Saturation filter softness
        /// </summary>
        public float ColorBleedHueSuppresionSaturationWidth = 0.2f;

        /// <summary>
        /// Limits brightness of self color bleeding
        /// </summary>
        public float ColorBleedHueSuppresionBrightness = 0.0f;

        /// <summary>
        /// Quality of color bleed: 1 - full quality, 2 - half of AO samples, 4 - quarter of AO samples
        /// </summary>
        public int ColorBleedQuality = 2;

        /// <summary>
        /// Controls how strong is suppresion of self-illumination by color bleeding
        /// </summary>
        public ColorBleedSelfOcclusionFixLevelType ColorBleedSelfOcclusionFixLevel = ColorBleedSelfOcclusionFixLevelType.Hard;

        /// <summary>
        /// Enables colorbleeding backfaces (faces behind illuminated surface)
        /// </summary>
        public bool GiBackfaces = false;

        #endregion

        #region Blur Settings

        /// <summary>
        /// Quality of blur step
        /// </summary>
        public BlurQualityType BlurQuality = VAOEffectCommandBuffer.BlurQualityType.Precise;

        /// <summary>
        /// Blur type: standard - 3x3 uniform weighted blur or enhanced - variable size multi-pass blur
        /// </summary>
        public BlurModeType BlurMode = VAOEffectCommandBuffer.BlurModeType.Enhanced;

        /// <summary>
        /// Size of enhanced blur in pixels
        /// </summary>
        public int EnhancedBlurSize = 5;

        /// <summary>
        /// Sharpnesss of enhanced blur (deviation of bell curve used for weighting blur samples)
        /// </summary>
        public float EnhancedBlurDeviation = 1.3f;

        #endregion

        /// <summary>
        /// Will draw only AO component to screen for debugging and fine-tuning purposes
        /// </summary>
        public bool OutputAOOnly = false;

        /// <summary>
        /// Choice of normals source - Unity's built-in GBuffer or calculation of normals from depth
        /// </summary>
        public NormalsSourceType NormalsSource = NormalsSourceType.GBuffer;

        #endregion

        #region VAO Enums

        public enum EffectMode
        {
            Simple = 1,
            ColorTint = 2,
            ColorBleed = 3
        }

        public enum LuminanceModeType
        {
            Luma = 1,
            HSVValue = 2
        }

        public enum GiBlurAmmount
        {
            Auto = 1,
            Less = 2,
            More = 3
        }

        public enum CullingPrepassModeType
        {
            Off = 0,
            Greedy = 1,
            Careful = 2
        }

        public enum AdaptiveSamplingType
        {
            Disabled = 0,
            EnabledAutomatic = 1,
            EnabledManual = 2
        }

        public enum BlurModeType
        {
            Disabled = 0,
            Basic = 1,
            Enhanced = 2
        }

        public enum BlurQualityType
        {
            Fast = 0,
            Precise = 1
        }

        public enum ColorBleedSelfOcclusionFixLevelType
        {
            Off = 0,
            Soft = 1,
            Hard = 2
        }

        public enum ScreenTextureFormat
        {
            Auto,
            ARGB32,
            ARGBHalf,
            ARGBFloat,
            Default,
            DefaultHDR
        }

        public enum FarPlaneSourceType
        {
            ProjectionParams,
            Camera
        }

        public enum DistanceFalloffModeType
        {
            Off = 0,
            Absolute = 1,
            Relative = 2
        }

        public enum VAOCameraEventType
        {
            AfterLighting, //< Apply To Lighting
            BeforeReflections, //< Apply To AO buffer and Lighting
            BeforeImageEffectsOpaque, //< Apply to Final Image
        }

        public enum HierarchicalBufferStateType
        {
            Off = 0,
            On = 1,
            Auto = 2
        }

        public enum NormalsSourceType
        {
            GBuffer = 1,
            Calculate = 2
        }

        public enum DetailQualityType
        {
            Medium = 1,
            High = 2
        }

        public enum AlgorithmType
        {
            StandardVAO = 1,
            RaycastAO = 2
        }

        private enum ShaderPass
        {
            CullingPrepass,
            MainPass,
            MainPassDoubleRadius,
            MainPassTripleRadius,
            MainPassDoubleRadiusHQ,
            MainPassTripleRadiusHQ
        }

        private enum ShaderFinalPass
        {
            StandardBlurUniform,
            StandardBlurUniformMultiplyBlend,
            StandardBlurUniformFast,
            StandardBlurUniformFastMultiplyBlend,
            EnhancedBlurFirstPass,
            EnhancedBlurSecondPass,
            EnhancedBlurSecondPassMultiplyBlend,
            Mixing,
            MixingMultiplyBlend,
            DownscaleDepthNormalsPass,
            Copy,
            BlendAfterLightingLog,
            TexCopyImageEffectSPSR,
            CalculateNormals,
            TexCopyTemporalSPSR
        }

        private enum ShaderBeforeReflectionsBlendPass
        {
            BlendBeforeReflections,
            BlendBeforeReflectionsLog
        }

        #endregion

        #region VAO Private Variables

        #region Performance Optimizations Settings

        /// <summary>
        /// This variable controls when will automatic control turn on hierarchical buffers.
        /// Lowering this number will cause turning on hierarchical buffers for lower radius settings.
        /// </summary>
        public float HierarchicalBufferPixelsPerLevel = 150.0f;

        private int CullingPrepassDownsamplingFactor = 8;

        private float AdaptiveQuality = 0.2f;
        private float AdaptiveMin = 0.0f;
        private float AdaptiveMax = -10.0f;

        #endregion

        #region Foldouts

        // Needs to be public so editor won't forget these

        public bool lumaFoldout = true;
        public bool colorBleedFoldout = false;
        public bool optimizationFoldout = true;
        public bool radiusLimitsFoldout = true;
        public bool pipelineFoldout = true;
        public bool blurFoldout = true;
        public bool aboutFoldout = false;

        #endregion

        #region Command Buffer Variables

        private Dictionary<CameraEvent, CommandBuffer> cameraEventsRegistered = new Dictionary<CameraEvent, CommandBuffer>();
        private bool isCommandBufferAlive = false;

        private Mesh screenQuad;

        private int destinationWidth;
        private int destinationHeight;

        private bool onDestroyCalled = false;

        #endregion

        #region Shader, Material, Camera

        public Shader vaoMainShader;
        public Shader vaoMainColorbleedShader;
        public Shader raycastMainShader;
        public Shader raycastMainColorbleedShader;
        public Shader vaoFinalPassShader;
        public Shader vaoBeforeReflectionsBlendShader;

        private Camera myCamera = null;
        private bool isSupported;
        private Material VAOMaterial;
        private Material VAOColorbleedMaterial;
        private Material VAOFinalPassMaterial;
        private Material RaycastMaterial;
        private Material RaycastColorbleedMaterial;
        private Material BeforeReflectionsBlendMaterial;

        #endregion

        #region Warning Flags

        public bool ForcedSwitchPerformed = false;
        public bool ForcedSwitchPerformedSinglePassStereo = false;
        public bool ForcedSwitchPerformedSinglePassStereoGBuffer = false;

        #endregion

        #region Previous controls values

        private int lastDownsampling;
        private int lastAlgorithm = 0;
        private AdaptiveSamplingType lastAdaptiveType;
        private CameraEvent? lastOverrideCameraEvent;
        private bool lastEnableTemporalFiltering;
        private int? lastOverrideWidth;
        private int? lastOverrideHeight;
        private int lastQuality;
        private CullingPrepassModeType lastcullingPrepassType;
        private int lastCullingPrepassDownsamplingFactor;
        private BlurModeType lastBlurMode;
        private BlurQualityType lastBlurQuality;
        private int lastMainPass;
        private EffectMode lastMode;
        private bool lastUseGBuffer;
        private bool lastOutputAOOnly;
        private CameraEvent lastCameraEvent;
        private bool lastIsHDR;
        private bool lastIsSPSR;
        private bool lastIsMPSR;
        
        private int lastScreenTextureWidth;
        private int lastScreenTextureHeight;
        private bool isHDR;
        public bool isSPSR;
        public bool isMPSR;
        private ScreenTextureFormat lastIntermediateScreenTextureFormat;
        private int lastCmdBufferEnhancedBlurSize;
        private bool lastHierarchicalBufferEnabled = false;

        #endregion

        #region VAO Private Data
        
        public bool historyReady = false;
        private Matrix4x4 previousCameraToWorldMatrix;
        private Matrix4x4 previousProjectionMatrix;
        private Matrix4x4 previousCameraToWorldMatrixLeft;
        private Matrix4x4 previousProjectionMatrixLeft;

        private Texture2D noiseTexture;
        private Texture2D temporalSamplesTexture;
        private int frameNumber = 0;
        private bool isEvenFrame = false;
        private Vector4[] adaptiveSamples = null;
        private Vector4[] carefulCache = null;

        private RenderTexture[] aoHistory = null;
        private int aoHistoryCurrentIdx = 0;

        private Vector4[] gaussian = null;
        private Vector4[] gaussianBuffer = new Vector4[17];

        // To prevent error with capping of large array to smaller size in Unity 5.4 - always use largest array filled with trailing zeros.
        private Vector4[] samplesLarge = new Vector4[80];
        int lastSamplesLength = 0;

        private int lastEnhancedBlurSize = 0;

        private float gaussianWeight = 0.0f;
        private float lastDeviation = 0.5f;


        #endregion

        public CameraEvent? OverrideCameraEvent = null;
        public int? OverrideWidth = null;
        public int? OverrideHeight = null;

        #endregion

        #region Unity Events

        void Start()
        {
            if (vaoMainShader == null) vaoMainShader = Shader.Find("Hidden/Wilberforce/VAOStandardShader");
            if (vaoMainColorbleedShader == null) vaoMainColorbleedShader = Shader.Find("Hidden/Wilberforce/VAOStandardColorbleedShader");
            if (raycastMainShader == null) raycastMainShader = Shader.Find("Hidden/Wilberforce/RaycastShader");
            if (raycastMainColorbleedShader == null) raycastMainColorbleedShader = Shader.Find("Hidden/Wilberforce/RaycastColorbleedShader");
            if (vaoFinalPassShader == null) vaoFinalPassShader = Shader.Find("Hidden/Wilberforce/VAOFinalPassShader");
            if (vaoBeforeReflectionsBlendShader == null) vaoBeforeReflectionsBlendShader = Shader.Find("Hidden/Wilberforce/VAOBeforeReflectionsBlendShader");

            if (vaoMainShader == null)
            {
                ReportError("Could not locate VAO Shader. Make sure there is 'VAO.shader' file added to the project.");
                isSupported = false;
                enabled = false;
                return;
            }

            t2d = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
            rt = new RenderTexture(512, 512, 128);

            // Do not report for colorbleed shaders to support at least standard mode on some platforms
            //if (vaoMainColorbleedShader == null)
            //{
            //    ReportError("Could not locate VAO Shader. Make sure there is 'VAO.shader' file added to the project.");
            //    isSupported = false;
            //    enabled = false;
            //    return;
            //}

            if (raycastMainShader == null)
            {
                ReportError("Could not locate VAO Shader. Make sure there is 'Raycast.shader' file added to the project.");
                isSupported = false;
                enabled = false;
                return;
            }

            if (vaoFinalPassShader == null)
            {
                ReportError("Could not locate VAO Shader. Make sure there is 'VAOFinalPassShader.shader' file added to the project.");
                isSupported = false;
                enabled = false;
                return;
            }

            if (vaoBeforeReflectionsBlendShader == null)
            {
                ReportError("Could not locate VAO Shader. Make sure there is 'VAOBeforeReflectionsBlendShader.shader' file added to the project.");
                isSupported = false;
                enabled = false;
                return;
            }


            if (!SystemInfo.supportsImageEffects || !SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.Depth) || SystemInfo.graphicsShaderLevel < 30)
            {
                if (!SystemInfo.supportsImageEffects) ReportError("System does not support image effects.");
                if (!SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.Depth)) ReportError("System does not support depth texture.");
                if (SystemInfo.graphicsShaderLevel < 30) ReportError("This effect needs at least Shader Model 3.0.");

                isSupported = false;
                enabled = false;
                return;
            }

            EnsureMaterials();

            if (!VAOMaterial || VAOMaterial.passCount != Enum.GetValues(typeof(ShaderPass)).Length)
            {
                ReportError("Could not create shader.");
                isSupported = false;
                enabled = false;
                return;
            }

            if (!RaycastMaterial || RaycastMaterial.passCount != Enum.GetValues(typeof(ShaderPass)).Length)
            {
                ReportError("Could not create shader.");
                isSupported = false;
                enabled = false;
                return;
            }

            if (!VAOFinalPassMaterial || VAOFinalPassMaterial.passCount != Enum.GetValues(typeof(ShaderFinalPass)).Length)
            {
                ReportError("Could not create shader.");
                isSupported = false;
                enabled = false;
                return;
            }

            if (!BeforeReflectionsBlendMaterial || BeforeReflectionsBlendMaterial.passCount != Enum.GetValues(typeof(ShaderBeforeReflectionsBlendPass)).Length)
            {
                ReportError("Could not create shader.");
                isSupported = false;
                enabled = false;
                return;
            }

            EnsureNoiseTexture();
            EnsureTemporalSamplesTexture();

            if (adaptiveSamples == null) adaptiveSamples = GenerateAdaptiveSamples();

            historyReady = false;
            isSupported = true;
        }

        void OnEnable()
        {
            this.myCamera = GetComponent<Camera>();
            TeardownCommandBuffer();
            historyReady = false;

            isSPSR = isCameraSPSR(myCamera);

            // See if there is post processing stuck
#if UNITY_EDITOR
            if (myCamera != null && (CommandBufferEnabled == false || myCamera.actualRenderingPath == RenderingPath.DeferredShading))
            {
                try
                {
#if !UNITY_2017_1_OR_NEWER
                    System.Reflection.Assembly asm = System.Reflection.Assembly.GetExecutingAssembly();
                    Type postStackType = asm.GetType("UnityEngine.PostProcessing.PostProcessingBehaviour");
                    var postStack = GetComponent(postStackType);
                    if (postStack != null)
                    {
                        if (!ForcedSwitchPerformed)
                        {
                            ReportWarning("Post Processing Stack Detected! Switching to command buffer pipeline and GBuffer inputs!");
                            CommandBufferEnabled = true;
                            UseGBuffer = true;
                            ForcedSwitchPerformed = true;
                        }
                    }
#endif
                }
                catch (Exception) { }

                // See if we are in single pass rendering
#if UNITY_5_5_OR_NEWER
                //if (myCamera.stereoEnabled && isSPSR)
                //{
                //    if (CommandBufferEnabled == false)
                //    {
                //        if (ForcedSwitchPerformedSinglePassStereo)
                //        {
                //            ReportWarning("You are running in single pass stereo mode! We recommend switching to command buffer pipeline if you encounter black screen problems.");
                //        }
                //        else
                //        {
                //            ReportWarning("You are running in single pass stereo mode! Switching to command buffer pipeline (recommended setting)!");
                //            CommandBufferEnabled = true;
                //            ForcedSwitchPerformedSinglePassStereo = true;
                //        }
                //    }

                //}
#endif

#if UNITY_2017_1_OR_NEWER
                if (myCamera.stereoEnabled
                    && isSPSR
                    && myCamera.actualRenderingPath == RenderingPath.DeferredShading)
                {
                    if (!ForcedSwitchPerformedSinglePassStereoGBuffer)
                    {
                        UseGBuffer = true;
                        ForcedSwitchPerformedSinglePassStereoGBuffer = true;
                    }
                }
#endif
            }
#endif

        }

        void OnValidate()
        {
            // Force parameters to be positive
            Radius = Mathf.Clamp(Radius, 0.001f, float.MaxValue);
            Power = Mathf.Clamp(Power, 0, float.MaxValue);
        }

        void OnPreRender()
        {
            EnsureVAOVersion();

            isEvenFrame = !isEvenFrame;
            if (myCamera.stereoEnabled && !isSPSR) TeardownCommandBuffer();

            bool forceDepthTexture = false;
            bool forceDepthNormalsTexture = false;

            if (NormalsSource == NormalsSourceType.Calculate)
                forceDepthTexture = true;

            if (Algorithm == AlgorithmType.RaycastAO)
                forceDepthTexture = true;

            DepthTextureMode currentMode = myCamera.depthTextureMode;
            if (myCamera.actualRenderingPath == RenderingPath.DeferredShading && UseGBuffer)
            {
                forceDepthTexture = true;
            }
            else
            {
                forceDepthNormalsTexture = true;
            }

            if (UsePreciseDepthBuffer && (myCamera.actualRenderingPath == RenderingPath.Forward || myCamera.actualRenderingPath == RenderingPath.VertexLit))
            {
                forceDepthTexture = true;
                forceDepthNormalsTexture = true;
            }

            if (forceDepthTexture)
            {
                if ((currentMode & DepthTextureMode.Depth) != DepthTextureMode.Depth)
                {
                    myCamera.depthTextureMode |= DepthTextureMode.Depth;
                }
            }

            if (forceDepthNormalsTexture)
            {
                if ((currentMode & DepthTextureMode.DepthNormals) != DepthTextureMode.DepthNormals)
                {
                    myCamera.depthTextureMode |= DepthTextureMode.DepthNormals;
                }
            }

            if (EnableTemporalFiltering)
            {
                if ((currentMode & DepthTextureMode.MotionVectors) != DepthTextureMode.MotionVectors)
                {
                    myCamera.depthTextureMode |= DepthTextureMode.MotionVectors;
                }
            }

            if (!(myCamera.stereoEnabled && !isSPSR) || !isEvenFrame)
            {
                frameNumber++;
                aoHistoryCurrentIdx++;
            }

            if (frameNumber > 3) frameNumber = 0;
            if (aoHistoryCurrentIdx > 1) aoHistoryCurrentIdx = 0;

            EnsureMaterials();
            EnsureNoiseTexture();
            EnsureTemporalSamplesTexture();

            TrySetUniforms();
            EnsureCommandBuffer(CheckSettingsChanges());
        }

        void OnPostRender()
        {
            historyReady = true;

            if (myCamera.stereoEnabled && !isSPSR)
            {
                if (myCamera.stereoActiveEye == Camera.MonoOrStereoscopicEye.Left)
                {
                    previousCameraToWorldMatrixLeft = myCamera.cameraToWorldMatrix;
                    previousProjectionMatrixLeft = myCamera.projectionMatrix;
                } else
                {
                    previousCameraToWorldMatrix = myCamera.cameraToWorldMatrix;
                    previousProjectionMatrix = myCamera.projectionMatrix;
                }

            }
            else
            {
                previousCameraToWorldMatrix = myCamera.cameraToWorldMatrix;
                previousProjectionMatrix = myCamera.projectionMatrix;
            }

#if UNITY_5_6_OR_NEWER

            if (myCamera == null || myCamera.activeTexture == null) return;

            // Check if cmd. buffer was created with correct target texture sizes and rebuild if necessary
            if (this.destinationWidth != myCamera.activeTexture.width || this.destinationHeight != myCamera.activeTexture.height || !isCommandBufferAlive)
            {
                this.destinationWidth = myCamera.activeTexture.width;
                this.destinationHeight = myCamera.activeTexture.height;

                TeardownCommandBuffer();
                EnsureCommandBuffer();
            }
            else
            {
                // Remember destination texture dimensions for use in command buffer (there are different values in camera.pixelWidth/Height which do not work in Single pass stereo)
                this.destinationWidth = myCamera.activeTexture.width;
                this.destinationHeight = myCamera.activeTexture.height;
            }
#endif

        }

        void OnDisable()
        {
            TeardownCommandBuffer();
        }

        void OnDestroy()
        {
            TeardownCommandBuffer();
            onDestroyCalled = true;
        }

        #endregion

        #region Image Effect Implementation

        ShaderPass GetMainPass(AlgorithmType algorithm, bool doQuarterRadius, bool doHalfRadius)
        {
            if (!doQuarterRadius && !doHalfRadius)
            {
                return ShaderPass.MainPass;
            }

            if (!doQuarterRadius && doHalfRadius)
            {
                return DetailQuality == DetailQualityType.Medium ? ShaderPass.MainPassDoubleRadius : ShaderPass.MainPassDoubleRadiusHQ;
            }

            return DetailQuality == DetailQualityType.Medium ? ShaderPass.MainPassTripleRadius : ShaderPass.MainPassTripleRadiusHQ;
        }


        RenderTexture rt;
        //RenderTexture rt;
        Texture2D t2d;
        int num = 0;


        //void Start()
        //{
        //    //GetComponent<Camera>().targetTexture = rt;
        //    //GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.DepthNormals;

        //    t2d = new Texture2D(512, 512, TextureFormat.RGBAFloat, false);
        //    rt = new RenderTexture(512, 512, 128);

        //}


        void Update()
        {
            GetComponent<Camera>().targetTexture = null;


            if (Input.GetKeyDown(KeyCode.F))
            {
                //_material.SetFloat("_Radius", 0);
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

                File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-vao++.exr", bytes);

                //File.ReadAllBytes();
                //File.WriteAllBytes(".//screenshot" + "//" + num.ToString() + "-ao.exr", bytes);


                Debug.Log("当前截图序号为：" + num.ToString());
                num++;
            }
        }

        private Material getAlgorithmMaterial()
        {

            if (Mode == EffectMode.ColorBleed)
                return Algorithm == AlgorithmType.RaycastAO ? RaycastColorbleedMaterial : VAOColorbleedMaterial;
            else
                return Algorithm == AlgorithmType.RaycastAO ? RaycastMaterial : VAOMaterial;

        }

        //[ImageEffectOpaque]
        protected void PerformOnRenderImage(RenderTexture source, RenderTexture destination)
        {
            if (!isSupported || !vaoMainShader.isSupported || !raycastMainShader.isSupported || !vaoFinalPassShader.isSupported || !vaoBeforeReflectionsBlendShader.isSupported)
            {
                enabled = false;
                return;
            }

            if (CommandBufferEnabled)
            {
                return; //< Return here, drawing will be done in command buffer
            }
            else
            {
                TeardownCommandBuffer();
            }

            int screenTextureWidth = source.width / Downsampling;
            int screenTextureHeight = source.height / Downsampling;

            if (OverrideWidth.HasValue) screenTextureWidth = OverrideWidth.Value;
            if (OverrideHeight.HasValue) screenTextureHeight = OverrideHeight.Value;

            if (screenTextureWidth != lastScreenTextureWidth ||
                screenTextureHeight != lastScreenTextureHeight)
            {
                releaseAoHistory();

                lastScreenTextureWidth = screenTextureWidth;
                lastScreenTextureHeight = screenTextureHeight;
            }

            RenderTexture downscaled2Texture = null;
            RenderTexture downscaled4Texture = null;
            Material algorithmMaterial = getAlgorithmMaterial();

            if (HierarchicalBufferEnabled)
            {
                RenderTextureFormat hBufferFormat = RenderTextureFormat.RHalf;
                if (Mode == EffectMode.ColorBleed) hBufferFormat = RenderTextureFormat.ARGBHalf;

                downscaled2Texture = RenderTexture.GetTemporary(source.width / 2, source.height / 2, 0, hBufferFormat);
                downscaled2Texture.filterMode = FilterMode.Bilinear;
                downscaled4Texture = RenderTexture.GetTemporary(source.width / 4, source.height / 4, 0, hBufferFormat);
                downscaled4Texture.filterMode = FilterMode.Bilinear;

                Graphics.Blit(null, downscaled2Texture, VAOFinalPassMaterial, (int)ShaderFinalPass.DownscaleDepthNormalsPass);
                DoShaderBlitCopy(downscaled2Texture, downscaled4Texture);

                if (downscaled2Texture != null) algorithmMaterial.SetTexture("depthNormalsTexture2", downscaled2Texture);
                if (downscaled4Texture != null) algorithmMaterial.SetTexture("depthNormalsTexture4", downscaled4Texture);
            }

            // Create temporary texture for AO
            RenderTextureFormat aoTextureFormat = RenderTextureFormat.RGHalf;

            if (Mode == EffectMode.ColorBleed)
            {
                aoTextureFormat = isHDR ? RenderTextureFormat.DefaultHDR : RenderTextureFormat.Default;
                algorithmMaterial.SetTexture("cbInputTex", source);
            }

            RenderTexture vaoTexture = RenderTexture.GetTemporary(screenTextureWidth, screenTextureHeight, 0, aoTextureFormat);
            vaoTexture.filterMode = FilterMode.Bilinear;

            algorithmMaterial.SetTexture("noiseTexture", noiseTexture);
            algorithmMaterial.SetTexture("temporalSamples", temporalSamplesTexture);

            // Culling pre-pass
            RenderTexture cullingPrepassTexture = null;
            RenderTexture cullingPrepassTextureHalfRes = null;
            if (CullingPrepassMode != CullingPrepassModeType.Off && !EnableTemporalFiltering)
            {
                RenderTextureFormat prepassFormat = RenderTextureFormat.R8;

                cullingPrepassTexture = RenderTexture.GetTemporary(source.width / CullingPrepassDownsamplingFactor, source.height / CullingPrepassDownsamplingFactor, 0, prepassFormat);
                cullingPrepassTexture.filterMode = FilterMode.Bilinear;
                cullingPrepassTextureHalfRes = RenderTexture.GetTemporary(source.width / (CullingPrepassDownsamplingFactor * 2), source.height / (CullingPrepassDownsamplingFactor * 2), 0, prepassFormat);
                cullingPrepassTextureHalfRes.filterMode = FilterMode.Bilinear;

                Graphics.Blit(source, cullingPrepassTexture, algorithmMaterial, (int)ShaderPass.CullingPrepass);
                DoShaderBlitCopy(cullingPrepassTexture, cullingPrepassTextureHalfRes);
            }

            // Main pass
            if (cullingPrepassTextureHalfRes != null) algorithmMaterial.SetTexture("cullingPrepassTexture", cullingPrepassTextureHalfRes);

            float DetailAmount = Algorithm == AlgorithmType.StandardVAO ? DetailAmountVAO : DetailAmountRaycast;

            if (EnableTemporalFiltering)
            {
                prepareAoHistory(screenTextureWidth, screenTextureHeight);
                int baseIdx = 0;

                if (myCamera.stereoEnabled && !isSPSR)
                {
                    baseIdx = isEvenFrame ? 0 : aoHistory.Length / 2;
                }

                RenderBuffer[] renderBuffer;

                if (Mode == EffectMode.ColorBleed)
                {
                    renderBuffer = new RenderBuffer[5];
                    renderBuffer[0] = (aoHistory[baseIdx + (4 * aoHistoryCurrentIdx)] as RenderTexture).colorBuffer;
                    renderBuffer[1] = (vaoTexture as RenderTexture).colorBuffer;
                    renderBuffer[2] = (aoHistory[baseIdx + (4 * aoHistoryCurrentIdx) + 1] as RenderTexture).colorBuffer;
                    renderBuffer[3] = (aoHistory[baseIdx + (4 * aoHistoryCurrentIdx) + 2] as RenderTexture).colorBuffer;
                    renderBuffer[4] = (aoHistory[baseIdx + (4 * aoHistoryCurrentIdx) + 3] as RenderTexture).colorBuffer;

                    algorithmMaterial.SetTexture("aoHistoryTexture", aoHistory[baseIdx + (4 * (1 - aoHistoryCurrentIdx))]);
                    algorithmMaterial.SetTexture("giHistoryTexture", aoHistory[baseIdx + (4 * (1 - aoHistoryCurrentIdx)) + 1]);
                    algorithmMaterial.SetTexture("gi2HistoryTexture", aoHistory[baseIdx + (4 * (1 - aoHistoryCurrentIdx)) + 2]);
                    algorithmMaterial.SetTexture("gi3HistoryTexture", aoHistory[baseIdx + (4 * (1 - aoHistoryCurrentIdx)) + 3]);
                }
                else
                {
                    renderBuffer = new RenderBuffer[2];
                    renderBuffer[0] = (aoHistory[baseIdx + aoHistoryCurrentIdx] as RenderTexture).colorBuffer;
                    renderBuffer[1] = (vaoTexture as RenderTexture).colorBuffer;
                    algorithmMaterial.SetTexture("aoHistoryTexture", aoHistory[baseIdx + (1 - aoHistoryCurrentIdx)]);
                }

                Graphics.SetRenderTarget(renderBuffer, (destination as RenderTexture).depthBuffer);

                Graphics.Blit(source, algorithmMaterial, (int)GetMainPass(Algorithm, DetailAmount > 0.5f, DetailAmount > 0.0f));
            }
            else
            {
                Graphics.Blit(source, vaoTexture, algorithmMaterial, (int)GetMainPass(Algorithm, DetailAmount > 0.5f, DetailAmount > 0.0f));
            }
            VAOFinalPassMaterial.SetTexture("textureAO", vaoTexture);

            if (BlurMode != BlurModeType.Disabled)
            {
                int blurTextureWidth = source.width;
                int blurTextureHeight = source.height;

                if (BlurQuality == BlurQualityType.Fast)
                {
                    blurTextureHeight /= 2;
                }

                // Blur pass
                if (BlurMode == BlurModeType.Enhanced)
                {
                    RenderTexture tempTexture = RenderTexture.GetTemporary(blurTextureWidth, blurTextureHeight, 0, aoTextureFormat);
                    tempTexture.filterMode = FilterMode.Bilinear;

                    Graphics.Blit(null, tempTexture, VAOFinalPassMaterial, (int)ShaderFinalPass.EnhancedBlurFirstPass);

                    VAOFinalPassMaterial.SetTexture("textureAO", tempTexture);
                    Graphics.Blit(source, destination, VAOFinalPassMaterial, (int)ShaderFinalPass.EnhancedBlurSecondPass);

                    RenderTexture.ReleaseTemporary(tempTexture);
                }
                else
                {
                    int uniformBlurPass = (BlurQuality == BlurQualityType.Fast) ? (int)ShaderFinalPass.StandardBlurUniformFast : (int)ShaderFinalPass.StandardBlurUniform;

                    Graphics.Blit(source, destination, VAOFinalPassMaterial, uniformBlurPass);
                }
            }
            else
            {
                // Mixing pass
                Graphics.Blit(source, destination, VAOFinalPassMaterial, (int)ShaderFinalPass.Mixing);
            }

            // Cleanup
            if (vaoTexture != null) RenderTexture.ReleaseTemporary(vaoTexture);
            if (cullingPrepassTexture != null) RenderTexture.ReleaseTemporary(cullingPrepassTexture);
            if (cullingPrepassTextureHalfRes != null) RenderTexture.ReleaseTemporary(cullingPrepassTextureHalfRes);
            if (downscaled2Texture != null) RenderTexture.ReleaseTemporary(downscaled2Texture);
            if (downscaled4Texture != null) RenderTexture.ReleaseTemporary(downscaled4Texture);
        }

        #endregion

        #region Command Buffer Implementation

        private void EnsureCommandBuffer(bool settingsDirty = false)
        {
            if ((!settingsDirty && isCommandBufferAlive) || !CommandBufferEnabled) return;
            if (onDestroyCalled) return;

            try
            {
                CreateCommandBuffer();
                lastCameraEvent = GetCameraEvent(VaoCameraEvent);
                isCommandBufferAlive = true;
            }
            catch (Exception ex)
            {
                ReportError("There was an error while trying to create command buffer. " + ex.Message);
            }
        }

        private void CreateCommandBuffer()
        {
            CommandBuffer commandBuffer;

            VAOMaterial = null;
            VAOColorbleedMaterial = null;
            VAOFinalPassMaterial = null;
            RaycastMaterial = null;
            RaycastColorbleedMaterial = null;
            BeforeReflectionsBlendMaterial = null;

            EnsureMaterials();

            TrySetUniforms();

            CameraEvent cameraEvent = GetCameraEvent(VaoCameraEvent);

            if (cameraEventsRegistered.TryGetValue(cameraEvent, out commandBuffer))
            {
                commandBuffer.Clear();
            }
            else
            {
                commandBuffer = new CommandBuffer();
                myCamera.AddCommandBuffer(cameraEvent, commandBuffer);

                commandBuffer.name = "Volumetric Ambient Occlusion";

                // Register
                cameraEventsRegistered[cameraEvent] = commandBuffer;
            }

            Material algorithmMaterial = getAlgorithmMaterial();

            bool isHwBlending = (!OutputAOOnly && Mode != EffectMode.ColorBleed);

            RenderTargetIdentifier targetTexture = BuiltinRenderTextureType.CameraTarget;
            int? emissionTexture = null;
            int? occlusionTexture = null;

            int cameraWidth = this.destinationWidth;
            int cameraHeight = this.destinationHeight;

            if (cameraWidth <= 0) cameraWidth = myCamera.pixelWidth;
            if (cameraHeight <= 0) cameraHeight = myCamera.pixelHeight;

            if (OverrideWidth.HasValue) cameraWidth = OverrideWidth.Value;
            if (OverrideHeight.HasValue) cameraHeight = OverrideHeight.Value;

            int screenTextureWidth = cameraWidth / Downsampling;
            int screenTextureHeight = cameraHeight / Downsampling;

            if (!OutputAOOnly)
            {
                if (!isHDR)
                {
                    if (cameraEvent == CameraEvent.AfterLighting || cameraEvent == CameraEvent.BeforeReflections)
                    {
                        targetTexture = BuiltinRenderTextureType.GBuffer3; //< Emission/lighting buffer

                        emissionTexture = Shader.PropertyToID("emissionTextureRT");
                        commandBuffer.GetTemporaryRT(emissionTexture.Value, cameraWidth, cameraHeight, 0, FilterMode.Bilinear, RenderTextureFormat.ARGB2101010, RenderTextureReadWrite.Linear);

                        // Make a copy of emission buffer for blending
                        commandBuffer.Blit(BuiltinRenderTextureType.GBuffer3, emissionTexture.Value, VAOFinalPassMaterial, (int)ShaderFinalPass.Copy);

                        commandBuffer.SetGlobalTexture("emissionTexture", emissionTexture.Value);

                        isHwBlending = false;
                    }
                }

                if (cameraEvent == CameraEvent.BeforeReflections || (cameraEvent == CameraEvent.AfterLighting && !isHDR && isSPSR))
                {
                    occlusionTexture = Shader.PropertyToID("occlusionTextureRT");
                    commandBuffer.GetTemporaryRT(occlusionTexture.Value, cameraWidth, cameraHeight, 0, FilterMode.Bilinear, RenderTextureFormat.RHalf, RenderTextureReadWrite.Linear);
                    commandBuffer.SetGlobalTexture("occlusionTexture", occlusionTexture.Value);

                    isHwBlending = false;
                }
            }

            int? screenTexture = null;
            if (Mode == EffectMode.ColorBleed)
            {
                RenderTextureFormat screenTextureFormat = GetRenderTextureFormat(IntermediateScreenTextureFormat, isHDR);

                screenTexture = Shader.PropertyToID("screenTextureRT");
                commandBuffer.GetTemporaryRT(screenTexture.Value, cameraWidth, cameraHeight, 0, FilterMode.Bilinear, screenTextureFormat, RenderTextureReadWrite.Linear);

                // Remember input
                commandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, screenTexture.Value);
            }

            int vaoTexture = Shader.PropertyToID("vaoTextureRT");
            RenderTextureFormat aoTextureFormat = RenderTextureFormat.RGHalf;

            if (Mode == EffectMode.ColorBleed)
            {
                aoTextureFormat = isHDR ? RenderTextureFormat.DefaultHDR : RenderTextureFormat.Default;
                    commandBuffer.SetGlobalTexture("cbInputTex", screenTexture.Value);
            }

            commandBuffer.GetTemporaryRT(vaoTexture, screenTextureWidth, screenTextureHeight, 0, FilterMode.Bilinear, aoTextureFormat, RenderTextureReadWrite.Linear);

            int? cullingPrepassTexture = null;
            int? cullingPrepassTextureHalfRes = null;

            // Prepare hierarchical buffers
            int? downscaled2Texture = null;
            int? downscaled4Texture = null;

            if (HierarchicalBufferEnabled)
            {
                RenderTextureFormat hBufferFormat = RenderTextureFormat.RHalf;
                if (Mode == EffectMode.ColorBleed) hBufferFormat = RenderTextureFormat.ARGBHalf;

                downscaled2Texture = Shader.PropertyToID("downscaled2TextureRT");
                downscaled4Texture = Shader.PropertyToID("downscaled4TextureRT");
                commandBuffer.GetTemporaryRT(downscaled2Texture.Value, cameraWidth / 2, cameraHeight / 2, 0, FilterMode.Bilinear, hBufferFormat, RenderTextureReadWrite.Linear);
                commandBuffer.GetTemporaryRT(downscaled4Texture.Value, cameraWidth / 4, cameraHeight / 4, 0, FilterMode.Bilinear, hBufferFormat, RenderTextureReadWrite.Linear);

                commandBuffer.Blit((Texture)null, downscaled2Texture.Value, VAOFinalPassMaterial, (int)ShaderFinalPass.DownscaleDepthNormalsPass);
                commandBuffer.Blit(downscaled2Texture.Value, downscaled4Texture.Value);

                if (downscaled2Texture != null) commandBuffer.SetGlobalTexture("depthNormalsTexture2", downscaled2Texture.Value);
                if (downscaled4Texture != null) commandBuffer.SetGlobalTexture("depthNormalsTexture4", downscaled4Texture.Value);
            }

            // Culling pre-pass
            if (CullingPrepassMode != CullingPrepassModeType.Off && !EnableTemporalFiltering)
            {
                cullingPrepassTexture = Shader.PropertyToID("cullingPrepassTextureRT");
                cullingPrepassTextureHalfRes = Shader.PropertyToID("cullingPrepassTextureHalfResRT");

                RenderTextureFormat prepassFormat = RenderTextureFormat.R8;

                commandBuffer.GetTemporaryRT(cullingPrepassTexture.Value, screenTextureWidth / CullingPrepassDownsamplingFactor, screenTextureHeight / CullingPrepassDownsamplingFactor, 0, FilterMode.Bilinear, prepassFormat, RenderTextureReadWrite.Linear);
                commandBuffer.GetTemporaryRT(cullingPrepassTextureHalfRes.Value, screenTextureWidth / (CullingPrepassDownsamplingFactor * 2), screenTextureHeight / (CullingPrepassDownsamplingFactor * 2), 0, FilterMode.Bilinear, prepassFormat, RenderTextureReadWrite.Linear);

                if (Mode == EffectMode.ColorBleed)
                {
                    commandBuffer.Blit(screenTexture.Value, cullingPrepassTexture.Value, algorithmMaterial, (int)ShaderPass.CullingPrepass);
                }
                else
                {
                    commandBuffer.Blit(targetTexture, cullingPrepassTexture.Value, algorithmMaterial, (int)ShaderPass.CullingPrepass);
                }
                commandBuffer.Blit(cullingPrepassTexture.Value, cullingPrepassTextureHalfRes.Value);

                commandBuffer.SetGlobalTexture("cullingPrepassTexture", cullingPrepassTextureHalfRes.Value);
            }

            // Main pass
            commandBuffer.SetGlobalTexture("noiseTexture", noiseTexture);
            commandBuffer.SetGlobalTexture("temporalSamples", temporalSamplesTexture);

            float DetailAmount = Algorithm == AlgorithmType.StandardVAO ? DetailAmountVAO : DetailAmountRaycast;
            int mainPass = (int)GetMainPass(Algorithm, DetailAmount > 0.5f, DetailAmount > 0.0f);

            if (EnableTemporalFiltering)
            {
                prepareAoHistory(screenTextureWidth, screenTextureHeight);

                int baseIdx = 0;
                if (myCamera.stereoEnabled && !isSPSR)
                {
                    baseIdx = isEvenFrame ? 0 : aoHistory.Length / 2;
                }

                if (Mode == EffectMode.ColorBleed)
                {
                    commandBuffer.SetRenderTarget(new RenderTargetIdentifier[]
                     {
                        aoHistory[baseIdx + 4],
                        vaoTexture,
                        aoHistory[baseIdx + 4 + 1],
                        aoHistory[baseIdx + 4 + 2],
                        aoHistory[baseIdx + 4 + 3]
                     }, vaoTexture);

                    commandBuffer.SetGlobalTexture("aoHistoryTexture", aoHistory[baseIdx]);
                    commandBuffer.SetGlobalTexture("giHistoryTexture", aoHistory[baseIdx + 1]);
                    commandBuffer.SetGlobalTexture("gi2HistoryTexture", aoHistory[baseIdx + 2]);
                    commandBuffer.SetGlobalTexture("gi3HistoryTexture", aoHistory[baseIdx + 3]);
                }
                else
                {
                    commandBuffer.SetGlobalTexture("aoHistoryTexture", aoHistory[baseIdx]);

                    commandBuffer.SetRenderTarget(new RenderTargetIdentifier[]
                     {
                        aoHistory[baseIdx + 1],
                        vaoTexture
                     }, vaoTexture);
                }
                
                commandBuffer.DrawMesh(GetScreenQuad(), Matrix4x4.identity, algorithmMaterial, 0, mainPass);

                if (Mode == EffectMode.ColorBleed)
                {
                    if (isSPSR)
                    {
                        commandBuffer.SetGlobalTexture("temporalTexCopySource", aoHistory[baseIdx + 4]);
                        commandBuffer.Blit(aoHistory[baseIdx + 4], aoHistory[baseIdx + 0], VAOFinalPassMaterial, (int)ShaderFinalPass.TexCopyTemporalSPSR);

                        commandBuffer.SetGlobalTexture("temporalTexCopySource", aoHistory[baseIdx + 5]);
                        commandBuffer.Blit(aoHistory[baseIdx + 5], aoHistory[baseIdx + 1], VAOFinalPassMaterial, (int)ShaderFinalPass.TexCopyTemporalSPSR);

                        commandBuffer.SetGlobalTexture("temporalTexCopySource", aoHistory[baseIdx + 6]);
                        commandBuffer.Blit(aoHistory[baseIdx + 6], aoHistory[baseIdx + 2], VAOFinalPassMaterial, (int)ShaderFinalPass.TexCopyTemporalSPSR);

                        commandBuffer.SetGlobalTexture("temporalTexCopySource", aoHistory[baseIdx + 7]);
                        commandBuffer.Blit(aoHistory[baseIdx + 7], aoHistory[baseIdx + 3], VAOFinalPassMaterial, (int)ShaderFinalPass.TexCopyTemporalSPSR);

                    }
                    else
                    {
                        commandBuffer.Blit(aoHistory[baseIdx + 4], aoHistory[baseIdx + 0]);
                        commandBuffer.Blit(aoHistory[baseIdx + 1 + 4], aoHistory[baseIdx + 1]);
                        commandBuffer.Blit(aoHistory[baseIdx + 2 + 4], aoHistory[baseIdx + 2]);
                        commandBuffer.Blit(aoHistory[baseIdx + 3 + 4], aoHistory[baseIdx + 3]);
                    }
                }
                else
                {
                    if (isSPSR)
                    {
                        commandBuffer.SetGlobalTexture("temporalTexCopySource", aoHistory[baseIdx + 1]);
                        commandBuffer.Blit(aoHistory[baseIdx + 1], aoHistory[baseIdx + 0], VAOFinalPassMaterial, (int)ShaderFinalPass.TexCopyTemporalSPSR);
                    }
                    else
                    {
                        commandBuffer.Blit(aoHistory[baseIdx + 1], aoHistory[baseIdx + 0]);
                    }
                }
            }
            else
            {
                if (Mode == EffectMode.ColorBleed)
                {
                    commandBuffer.Blit(screenTexture.Value, vaoTexture, algorithmMaterial, mainPass);
                }
                else
                {
                    commandBuffer.Blit(targetTexture, vaoTexture, algorithmMaterial, mainPass);
                }
            }

            commandBuffer.SetGlobalTexture("textureAO", vaoTexture);

            if (BlurMode != BlurModeType.Disabled)
            {
                int blurTextureWidth = cameraWidth;
                int blurTextureHeight = cameraHeight;

                if (BlurQuality == BlurQualityType.Fast)
                {
                    blurTextureHeight /= 2;
                }

                // Blur pass
                if (BlurMode == BlurModeType.Enhanced)
                {
                    int tempTexture = Shader.PropertyToID("tempTextureRT");
                    commandBuffer.GetTemporaryRT(tempTexture, blurTextureWidth, blurTextureHeight, 0, FilterMode.Bilinear, aoTextureFormat, RenderTextureReadWrite.Linear);

                    commandBuffer.Blit(null, tempTexture, VAOFinalPassMaterial, (int)ShaderFinalPass.EnhancedBlurFirstPass);
                    commandBuffer.SetGlobalTexture("textureAO", tempTexture);

                    DoMixingBlit(commandBuffer, screenTexture, occlusionTexture, targetTexture, isHwBlending ? (int)ShaderFinalPass.EnhancedBlurSecondPassMultiplyBlend : (int)ShaderFinalPass.EnhancedBlurSecondPass, VAOFinalPassMaterial);

                    commandBuffer.ReleaseTemporaryRT(tempTexture);

                }
                else
                {
                    int uniformBlurPass = (BlurQuality == BlurQualityType.Fast) ? (int)ShaderFinalPass.StandardBlurUniformFast : (int)ShaderFinalPass.StandardBlurUniform;
                    int uniformBlurPassBlend = (BlurQuality == BlurQualityType.Fast) ? (int)ShaderFinalPass.StandardBlurUniformFastMultiplyBlend : (int)ShaderFinalPass.StandardBlurUniformMultiplyBlend;

                    DoMixingBlit(commandBuffer, screenTexture, occlusionTexture, targetTexture, isHwBlending ? uniformBlurPassBlend : uniformBlurPass, VAOFinalPassMaterial);
                }
            }
            else
            {
                // Mixing pass
                DoMixingBlit(commandBuffer, screenTexture, occlusionTexture, targetTexture, isHwBlending ? (int)ShaderFinalPass.MixingMultiplyBlend : (int)ShaderFinalPass.Mixing, VAOFinalPassMaterial);
            }

            if (cameraEvent == CameraEvent.BeforeReflections)
            {

                commandBuffer.SetRenderTarget(new RenderTargetIdentifier[]
                {
                    BuiltinRenderTextureType.GBuffer0,
                    targetTexture
                }, BuiltinRenderTextureType.GBuffer0);

                commandBuffer.DrawMesh(GetScreenQuad(), Matrix4x4.identity, BeforeReflectionsBlendMaterial, 0, isHDR ? (int)ShaderBeforeReflectionsBlendPass.BlendBeforeReflections : (int)ShaderBeforeReflectionsBlendPass.BlendBeforeReflectionsLog);
            }
            else if (cameraEvent == CameraEvent.AfterLighting && !isHDR && isSPSR)
            {
                commandBuffer.SetRenderTarget(targetTexture);
                commandBuffer.DrawMesh(GetScreenQuad(), Matrix4x4.identity, VAOFinalPassMaterial, 0, (int)ShaderFinalPass.BlendAfterLightingLog);
            }

            // Cleanup
            commandBuffer.ReleaseTemporaryRT(vaoTexture);
            if (screenTexture != null) commandBuffer.ReleaseTemporaryRT(screenTexture.Value);
            if (emissionTexture != null) commandBuffer.ReleaseTemporaryRT(emissionTexture.Value);
            if (occlusionTexture != null) commandBuffer.ReleaseTemporaryRT(occlusionTexture.Value);
            if (cullingPrepassTexture != null) commandBuffer.ReleaseTemporaryRT(cullingPrepassTexture.Value);
            if (cullingPrepassTextureHalfRes != null) commandBuffer.ReleaseTemporaryRT(cullingPrepassTextureHalfRes.Value);
            if (downscaled2Texture != null) commandBuffer.ReleaseTemporaryRT(downscaled2Texture.Value);
            if (downscaled4Texture != null) commandBuffer.ReleaseTemporaryRT(downscaled4Texture.Value);
        }

        private void TeardownCommandBuffer()
        {
            if (!isCommandBufferAlive) return;

            if (!(myCamera.stereoEnabled && !isSPSR)) releaseAoHistory();

            try
            {
                isCommandBufferAlive = false;

                if (myCamera != null)
                {
                    foreach (var e in cameraEventsRegistered)
                    {
                        myCamera.RemoveCommandBuffer(e.Key, e.Value);
                    }
                }

                cameraEventsRegistered.Clear();
                VAOMaterial = null;
                VAOFinalPassMaterial = null;
                RaycastMaterial = null;
                BeforeReflectionsBlendMaterial = null;
                EnsureMaterials();
            }
            catch (Exception ex)
            {
                ReportError("There was an error while trying to destroy command buffer. " + ex.Message);
            }
        }

        #region Command Buffer Utilities

        protected Mesh GetScreenQuad()
        {
            if (screenQuad == null)
            {
                screenQuad = new Mesh()
                {
                    vertices = new Vector3[] { new Vector3(-1, -1, 0), new Vector3(-1, 1, 0), new Vector3(1, 1, 0), new Vector3(1, -1, 0) },
                    triangles = new int[] { 0, 1, 2, 0, 2, 3 },
                    uv = new Vector2[] { new Vector2(0, 1), new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1) }
                };
            }

            return screenQuad;
        }

        private CameraEvent GetCameraEvent(VAOCameraEventType vaoCameraEvent)
        {

            if (this.OverrideCameraEvent.HasValue) return this.OverrideCameraEvent.Value;

            if (myCamera == null) return CameraEvent.BeforeImageEffectsOpaque;
            if (OutputAOOnly) return CameraEvent.BeforeImageEffectsOpaque;
            if (Mode == EffectMode.ColorBleed) return CameraEvent.BeforeImageEffectsOpaque;

            if (myCamera.actualRenderingPath != RenderingPath.DeferredShading)
            {
                return CameraEvent.BeforeImageEffectsOpaque;
            }

            switch (vaoCameraEvent)
            {
                case VAOCameraEventType.AfterLighting:
                    return CameraEvent.AfterLighting;
                case VAOCameraEventType.BeforeImageEffectsOpaque:
                    return CameraEvent.BeforeImageEffectsOpaque;
                case VAOCameraEventType.BeforeReflections:
                    return CameraEvent.BeforeReflections;
                default:
                    return CameraEvent.BeforeImageEffectsOpaque;
            }
        }

        private void DoShaderBlitCopy(Texture sourceTexture, RenderTexture destinationTexture)
        {
            if (isSPSR && !CommandBufferEnabled)
            {
                VAOFinalPassMaterial.SetTexture("texCopySource", sourceTexture);
                Graphics.Blit(sourceTexture, destinationTexture, VAOFinalPassMaterial, (int)ShaderFinalPass.TexCopyImageEffectSPSR);
            }
            else
            {
                Graphics.Blit(sourceTexture, destinationTexture);
            }
        }

        protected void DoMixingBlit(CommandBuffer commandBuffer, int? source, int? primaryTarget, RenderTargetIdentifier secondaryTarget, int pass, Material material)
        {
            if (primaryTarget.HasValue)
                DoBlit(commandBuffer, source, primaryTarget.Value, pass, material);
            else
                DoBlit(commandBuffer, source, secondaryTarget, pass, material);
        }

        protected void DoBlit(CommandBuffer commandBuffer, int? source, int target, int pass, Material material)
        {
            if (source.HasValue)
                commandBuffer.Blit(source.Value, target, material, pass);
            else
                commandBuffer.Blit((Texture)null, target, material, pass);
        }

        protected void DoBlit(CommandBuffer commandBuffer, int? source, RenderTargetIdentifier target, int pass, Material material)
        {
            if (source.HasValue)
                commandBuffer.Blit(source.Value, target, material, pass);
            else
                commandBuffer.Blit((Texture)null, target, material, pass);
        }

        #endregion

        #endregion

        #region Shader Utilities

        private void TrySetUniforms()
        {
            Material algorithmMaterial = getAlgorithmMaterial();

            if (algorithmMaterial == null) return;
            if (VAOFinalPassMaterial == null) return;
            if (BeforeReflectionsBlendMaterial == null) return;

            int screenTextureWidth = myCamera.pixelWidth / Downsampling;
            int screenTextureHeight = myCamera.pixelHeight / Downsampling;

            if (!EnableTemporalFiltering)
            {
                releaseAoHistory();
            }
                 
            switch (Quality)
            {
                case 2:
                    algorithmMaterial.SetInt("maxSamplesCount", 2);
                    algorithmMaterial.SetInt("samplesStartIndex", (64 + 32 + 16 + 8 + 4) / 2);
                    break;
                case 4:
                    algorithmMaterial.SetInt("maxSamplesCount", 4);
                    algorithmMaterial.SetInt("samplesStartIndex", (64 + 32 + 16 + 8) / 2);
                    break;
                case 8:
                    algorithmMaterial.SetInt("maxSamplesCount", 8);
                    algorithmMaterial.SetInt("samplesStartIndex", (64 + 32 + 16) / 2);
                    break;
                case 16:
                    algorithmMaterial.SetInt("maxSamplesCount", 16);
                    algorithmMaterial.SetInt("samplesStartIndex", (64 + 32) / 2);
                    break;
                case 32:
                    algorithmMaterial.SetInt("maxSamplesCount", 32);
                    algorithmMaterial.SetInt("samplesStartIndex", (64) / 2);
                    break;
                case 64:
                    algorithmMaterial.SetInt("maxSamplesCount", 64);
                    algorithmMaterial.SetInt("samplesStartIndex", 0);
                    break;
                default:
                    ReportError("Unsupported quality setting " + Quality + " encountered. Reverting to low setting");
                    // Reverting to low
                    algorithmMaterial.SetInt("maxSamplesCount", 16);
                    algorithmMaterial.SetInt("samplesStartIndex", (64 + 32) / 2);
                    Quality = 16;
                    break;
            }

            if (AdaptiveType != AdaptiveSamplingType.Disabled)
            {
                switch (Quality)
                {
                    case 64: AdaptiveQuality = 0.025f; break;
                    case 32: AdaptiveQuality = 0.025f; break;
                    case 16: AdaptiveQuality = 0.05f; break;
                    case 8: AdaptiveQuality = 0.1f; break;
                    case 4: AdaptiveQuality = 0.2f; break;
                    case 2: AdaptiveQuality = 0.4f; break;
                }
                if (AdaptiveType == AdaptiveSamplingType.EnabledManual)
                {
                    AdaptiveQuality *= AdaptiveQualityCoefficient;
                }
                else
                {
                    AdaptiveQualityCoefficient = 1.0f;
                }
            }

            AdaptiveMax = GetDepthForScreenSize(myCamera, AdaptiveQuality);


#if UNITY_EDITOR
            if (EnableTemporalFiltering && !historyReady)
            {
                algorithmMaterial.SetInt("maxSamplesCount", 64);
                algorithmMaterial.SetInt("samplesStartIndex", 0);
            }

            AdaptiveMax = GetDepthForScreenSize(myCamera, 0.005f);
#endif

            Vector2 texelSize = new Vector2(1.0f / screenTextureWidth, 1.0f / screenTextureHeight);
            float subPixelDepth = GetDepthForScreenSize(myCamera, Mathf.Max(texelSize.x, texelSize.y));

            bool mustFixSrspAfterLighting = GetCameraEvent(VaoCameraEvent) == CameraEvent.AfterLighting && isSPSR && !isHDR;

            algorithmMaterial.SetInt("noVertexTransform", (CommandBufferEnabled && EnableTemporalFiltering) ? 1 : 0);
            VAOFinalPassMaterial.SetInt("noVertexTransform", (CommandBufferEnabled && EnableTemporalFiltering) ? 1 : 0);
            algorithmMaterial.SetVector("inputTexDimensions", new Vector4(screenTextureWidth, screenTextureHeight, 0, 0));
            algorithmMaterial.SetInt("historyReady", (historyReady && aoHistory != null && aoHistory.Length > 0 && aoHistory[0] != null) ? 1 : 0);

            
            algorithmMaterial.SetInt("useUnityMotionVectors", (CommandBufferEnabled && GetCameraEvent(VaoCameraEvent) != CameraEvent.BeforeImageEffectsOpaque) ? 0 : 1);

#if UNITY_EDITOR
            algorithmMaterial.SetInt("enableReprojection", (EditorApplication.isPaused) ? 0 : 1);
#else
            algorithmMaterial.SetInt("enableReprojection", 1);
#endif
            algorithmMaterial.SetInt("normalsSource", (int)NormalsSource);
            VAOFinalPassMaterial.SetInt("normalsSource", (int)NormalsSource);
            
            algorithmMaterial.SetInt("isImageEffectMode", CommandBufferEnabled ? 0 : 1);
            VAOFinalPassMaterial.SetInt("isImageEffectMode", CommandBufferEnabled ? 0 : 1);
            BeforeReflectionsBlendMaterial.SetInt("isImageEffectMode", CommandBufferEnabled ? 0 : 1);

            int useSPSRFriendlyTransform = (isSPSR && !CommandBufferEnabled) ? 1 : 0;

            algorithmMaterial.SetInt("useSPSRFriendlyTransform", useSPSRFriendlyTransform);
            VAOFinalPassMaterial.SetInt("useSPSRFriendlyTransform", useSPSRFriendlyTransform);
            BeforeReflectionsBlendMaterial.SetInt("useSPSRFriendlyTransform", useSPSRFriendlyTransform);

            // Set shader uniforms
            if (myCamera.stereoEnabled && !isSPSR)
            {
                if (historyReady)
                {
                    if (myCamera.stereoActiveEye == Camera.MonoOrStereoscopicEye.Left)
                        algorithmMaterial.SetMatrix("motionVectorsMatrix", previousProjectionMatrixLeft * (previousCameraToWorldMatrixLeft.inverse * myCamera.cameraToWorldMatrix));
                    else
                        algorithmMaterial.SetMatrix("motionVectorsMatrix", previousProjectionMatrix * (previousCameraToWorldMatrix.inverse * myCamera.cameraToWorldMatrix));
                }
                else
                {
                    if (myCamera.stereoActiveEye == Camera.MonoOrStereoscopicEye.Left)
                        algorithmMaterial.SetMatrix("motionVectorsMatrix", myCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left));
                    else
                        algorithmMaterial.SetMatrix("motionVectorsMatrix", myCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right));
                }
            }
            else
            {
                if (historyReady)
                {
                    algorithmMaterial.SetMatrix("motionVectorsMatrix", previousProjectionMatrix * (previousCameraToWorldMatrix.inverse * myCamera.cameraToWorldMatrix));
                }
                else
                {
                    algorithmMaterial.SetMatrix("motionVectorsMatrix", myCamera.projectionMatrix);
                }
            }


            algorithmMaterial.SetMatrix("invProjMatrix", myCamera.projectionMatrix.inverse);
            VAOFinalPassMaterial.SetMatrix("invProjMatrix", myCamera.projectionMatrix.inverse);
            BeforeReflectionsBlendMaterial.SetMatrix("invProjMatrix", myCamera.projectionMatrix.inverse);

            Vector4 screenProjection = -0.5f * new Vector4(myCamera.projectionMatrix.m00, myCamera.projectionMatrix.m11, myCamera.projectionMatrix.m02, myCamera.projectionMatrix.m12);
            Vector4 screenUnprojection = new Vector4(myCamera.projectionMatrix.inverse.m00, myCamera.projectionMatrix.inverse.m11, myCamera.projectionMatrix.inverse.m23, 0.0f);
            Vector4 screenUnprojection2 = new Vector4(myCamera.projectionMatrix.inverse.m03, myCamera.projectionMatrix.inverse.m13, 0.0f, 1.0f / (myCamera.projectionMatrix.inverse.m32 + myCamera.projectionMatrix.inverse.m33));

            algorithmMaterial.SetVector("screenProjection", screenProjection);
            VAOFinalPassMaterial.SetVector("screenProjection", screenProjection);
            BeforeReflectionsBlendMaterial.SetVector("screenProjection", screenProjection);

            algorithmMaterial.SetVector("screenUnprojection", screenUnprojection);
            VAOFinalPassMaterial.SetVector("screenUnprojection", screenUnprojection);
            BeforeReflectionsBlendMaterial.SetVector("screenUnprojection", screenUnprojection);

            algorithmMaterial.SetVector("screenUnprojection2", screenUnprojection2);
            VAOFinalPassMaterial.SetVector("screenUnprojection2", screenUnprojection2);
            BeforeReflectionsBlendMaterial.SetVector("screenUnprojection2", screenUnprojection2);

            algorithmMaterial.SetFloat("halfRadiusSquared", (Radius * 0.5f) * (Radius * 0.5f));
            algorithmMaterial.SetFloat("halfRadius", Radius * 0.5f);
            algorithmMaterial.SetFloat("radius", Radius);
            algorithmMaterial.SetFloat("ssaoBias", SSAOBias);

            algorithmMaterial.SetInt("sampleCount", Quality);

#if UNITY_EDITOR
            if (EnableTemporalFiltering && !historyReady)
            {
                algorithmMaterial.SetInt("sampleCount", 64);
            }
#endif

            algorithmMaterial.SetInt("fourSamplesStartIndex", (64 + 32 + 16 + 8) / 2);
            algorithmMaterial.SetInt("eightSamplesStartIndex", (64 + 32 + 16) / 2);

            if (Algorithm == AlgorithmType.RaycastAO)
            {
                // Compensate for hemisphere volume
                algorithmMaterial.SetFloat("aoPower", Power * (Radius * 0.5f));
            }
            else
            {
                algorithmMaterial.SetFloat("aoPower", Power);
            }

            algorithmMaterial.SetFloat("aoPresence", Presence);
            algorithmMaterial.SetFloat("aoThickness", Thickness);
            algorithmMaterial.SetFloat("bordersIntensity", 1.0f - BordersIntensity);
            VAOFinalPassMaterial.SetFloat("giPresence", 1.0f - ColorBleedPresence);
            algorithmMaterial.SetFloat("LumaThreshold", LumaThreshold);
            VAOFinalPassMaterial.SetFloat("LumaThreshold", LumaThreshold);
            algorithmMaterial.SetFloat("LumaKneeWidth", LumaKneeWidth);
            VAOFinalPassMaterial.SetFloat("LumaKneeWidth", LumaKneeWidth);
            algorithmMaterial.SetFloat("LumaTwiceKneeWidthRcp", 1.0f / (LumaKneeWidth * 2.0f));
            VAOFinalPassMaterial.SetFloat("LumaTwiceKneeWidthRcp", 1.0f / (LumaKneeWidth * 2.0f));
            algorithmMaterial.SetFloat("LumaKneeLinearity", LumaKneeLinearity);
            VAOFinalPassMaterial.SetFloat("LumaKneeLinearity", LumaKneeLinearity);
            algorithmMaterial.SetInt("giBackfaces", GiBackfaces ? 0 : 1);
            algorithmMaterial.SetFloat("adaptiveMin", AdaptiveMin);
            algorithmMaterial.SetFloat("adaptiveMax", AdaptiveMax);

            algorithmMaterial.SetVector("texelSizeRcp", texelSize);
            VAOFinalPassMaterial.SetVector("texelSizeRcp", texelSize);

            VAOFinalPassMaterial.SetVector("texelSize", (BlurMode == BlurModeType.Basic && BlurQuality == BlurQualityType.Fast) ? texelSize * 0.5f : texelSize);

            VAOFinalPassMaterial.SetFloat("blurDepthThreshold", Radius);
            algorithmMaterial.SetInt("cullingPrepassMode", (int)(EnableTemporalFiltering ? CullingPrepassModeType.Off : CullingPrepassMode));
            algorithmMaterial.SetVector("cullingPrepassTexelSize", new Vector2(0.5f / (myCamera.pixelWidth / CullingPrepassDownsamplingFactor), 0.5f / (myCamera.pixelHeight / CullingPrepassDownsamplingFactor)));
            algorithmMaterial.SetInt("giSelfOcclusionFix", (int)ColorBleedSelfOcclusionFixLevel);
            algorithmMaterial.SetInt("adaptiveMode", (int)AdaptiveType);

            algorithmMaterial.SetInt("LumaMode", (int)LuminanceMode);
            VAOFinalPassMaterial.SetInt("LumaMode", (int)LuminanceMode);

            algorithmMaterial.SetFloat("cameraFarPlane", myCamera.farClipPlane);
            VAOFinalPassMaterial.SetFloat("cameraFarPlane", myCamera.farClipPlane);
            algorithmMaterial.SetInt("UseCameraFarPlane", FarPlaneSource == FarPlaneSourceType.Camera ? 1 : 0);
            VAOFinalPassMaterial.SetInt("UseCameraFarPlane", FarPlaneSource == FarPlaneSourceType.Camera ? 1 : 0);

            algorithmMaterial.SetFloat("maxRadiusEnabled", MaxRadiusEnabled ? 1 : 0);
            algorithmMaterial.SetFloat("maxRadiusCutoffDepth", GetDepthForScreenSize(myCamera, MaxRadius));
            algorithmMaterial.SetFloat("projMatrix11", myCamera.projectionMatrix.m11);
            algorithmMaterial.SetFloat("maxRadiusOnScreen", MaxRadius);
            VAOFinalPassMaterial.SetFloat("enhancedBlurSize", EnhancedBlurSize / 2);

            algorithmMaterial.SetInt("flipY", MustForceFlip(myCamera) ? 1 : 0);
            VAOFinalPassMaterial.SetInt("flipY", MustForceFlip(myCamera) ? 1 : 0);
            BeforeReflectionsBlendMaterial.SetInt("flipY", MustForceFlip(myCamera) ? 1 : 0);

            algorithmMaterial.SetInt("useGBuffer", ShouldUseGBuffer() ? 1 : 0);
            VAOFinalPassMaterial.SetInt("useGBuffer", ShouldUseGBuffer() ? 1 : 0);

            algorithmMaterial.SetInt("hierarchicalBufferEnabled", HierarchicalBufferEnabled ? 1 : 0);

            int hwBlendingEnabled = (CommandBufferEnabled && Mode != EffectMode.ColorBleed) && GetCameraEvent(VaoCameraEvent) != CameraEvent.BeforeReflections && !mustFixSrspAfterLighting ? 1 : 0;
            algorithmMaterial.SetInt("hwBlendingEnabled", hwBlendingEnabled);
            VAOFinalPassMaterial.SetInt("hwBlendingEnabled", hwBlendingEnabled);

            int useLogEmissiveBuffer = (CommandBufferEnabled && !isHDR && GetCameraEvent(VaoCameraEvent) == CameraEvent.AfterLighting && !isSPSR) ? 1 : 0;
            algorithmMaterial.SetInt("useLogEmissiveBuffer", useLogEmissiveBuffer);
            VAOFinalPassMaterial.SetInt("useLogEmissiveBuffer", useLogEmissiveBuffer);

            int useLogBufferInput = (CommandBufferEnabled && !isHDR && (GetCameraEvent(VaoCameraEvent) == CameraEvent.AfterLighting || GetCameraEvent(VaoCameraEvent) == CameraEvent.BeforeReflections)) ? 1 : 0;
            algorithmMaterial.SetInt("useLogBufferInput", useLogBufferInput);
            VAOFinalPassMaterial.SetInt("useLogBufferInput", useLogBufferInput);

            algorithmMaterial.SetInt("outputAOOnly", OutputAOOnly ? 1 : 0);
            VAOFinalPassMaterial.SetInt("outputAOOnly", OutputAOOnly ? 1 : 0);

            algorithmMaterial.SetInt("isLumaSensitive", IsLumaSensitive ? 1 : 0);
            VAOFinalPassMaterial.SetInt("isLumaSensitive", IsLumaSensitive ? 1 : 0);

            VAOFinalPassMaterial.SetInt("useFastBlur", BlurQuality == BlurQualityType.Fast ? 1 : 0);

            int useDedicatedDepthBuffer = ((UsePreciseDepthBuffer && (myCamera.actualRenderingPath == RenderingPath.Forward || myCamera.actualRenderingPath == RenderingPath.VertexLit))) ? 1 : 0;
            algorithmMaterial.SetInt("useDedicatedDepthBuffer", useDedicatedDepthBuffer);
            VAOFinalPassMaterial.SetInt("useDedicatedDepthBuffer", useDedicatedDepthBuffer);
            BeforeReflectionsBlendMaterial.SetInt("useDedicatedDepthBuffer", useDedicatedDepthBuffer);

            float quarterResDistance = GetDepthForScreenSize(myCamera, Mathf.Max(texelSize.x, texelSize.y) * HierarchicalBufferPixelsPerLevel * 2.0f);
            float halfResDistance = GetDepthForScreenSize(myCamera, Mathf.Max(texelSize.x, texelSize.y) * HierarchicalBufferPixelsPerLevel);

            quarterResDistance /= -myCamera.farClipPlane;
            halfResDistance /= -myCamera.farClipPlane;

            algorithmMaterial.SetFloat("quarterResBufferMaxDistance", quarterResDistance);
            algorithmMaterial.SetFloat("halfResBufferMaxDistance", halfResDistance);

            float DetailAmount = Algorithm == AlgorithmType.StandardVAO ? DetailAmountVAO : DetailAmountRaycast;

            algorithmMaterial.SetFloat("halfRadiusWeight", Mathf.Clamp01(DetailAmount * 2.0f));
            algorithmMaterial.SetFloat("quarterRadiusWeight", Mathf.Clamp01((DetailAmount - 0.5f) * 2.0f));

            algorithmMaterial.SetInt("minRadiusEnabled", (int)DistanceFalloffMode);
            algorithmMaterial.SetFloat("minRadiusCutoffDepth", DistanceFalloffMode == DistanceFalloffModeType.Relative ? Mathf.Abs(subPixelDepth) * -(DistanceFalloffStartRelative * DistanceFalloffStartRelative) : -DistanceFalloffStartAbsolute);
            algorithmMaterial.SetFloat("minRadiusSoftness", DistanceFalloffMode == DistanceFalloffModeType.Relative ? Mathf.Abs(subPixelDepth) * (DistanceFalloffSpeedRelative * DistanceFalloffSpeedRelative) : DistanceFalloffSpeedAbsolute);

            VAOFinalPassMaterial.SetInt("giSameHueAttenuationEnabled", ColorbleedHueSuppresionEnabled ? 1 : 0);
            VAOFinalPassMaterial.SetFloat("giSameHueAttenuationThreshold", ColorBleedHueSuppresionThreshold);
            VAOFinalPassMaterial.SetFloat("giSameHueAttenuationWidth", ColorBleedHueSuppresionWidth);
            VAOFinalPassMaterial.SetFloat("giSameHueAttenuationSaturationThreshold", ColorBleedHueSuppresionSaturationThreshold);
            VAOFinalPassMaterial.SetFloat("giSameHueAttenuationSaturationWidth", ColorBleedHueSuppresionSaturationWidth);
            VAOFinalPassMaterial.SetFloat("giSameHueAttenuationBrightness", ColorBleedHueSuppresionBrightness);
            algorithmMaterial.SetFloat("subpixelRadiusCutoffDepth", Mathf.Min(0.99f, subPixelDepth / -myCamera.farClipPlane));

            algorithmMaterial.SetVector("noiseTexelSizeRcp", new Vector2(screenTextureWidth / 3.0f, screenTextureHeight / 3.0f));

            if (temporalSamplesTexture != null)
            {
                Vector2 temporalSampelsTexSizeRcp = new Vector2(1.0f / ((float)temporalSamplesTexture.width), 1.0f / ((float)temporalSamplesTexture.height));
                algorithmMaterial.SetVector("temporalTexelSizeRcp", new Vector4(temporalSampelsTexSizeRcp.x, temporalSampelsTexSizeRcp.y, temporalSampelsTexSizeRcp.x * 0.5f, temporalSampelsTexSizeRcp.y * 0.5f));
            }

            algorithmMaterial.SetInt("frameNumber", frameNumber);
            
            SetKeywords(algorithmMaterial, "DISABLE_TEMPORAL_ACCUMULATION", "ENABLE_TEMPORAL_ACCUMULATION", EnableTemporalFiltering);

            if (Quality == 4 || (Quality == 8 && (ColorBleedQuality == 2 || ColorBleedQuality == 4)))
            {
                VAOFinalPassMaterial.SetInt("giBlur", (int)GiBlurAmmount.More);
            }
            else
            {
                VAOFinalPassMaterial.SetInt("giBlur", (int)GiBlurAmmount.Less);
            }

            if (Mode == EffectMode.ColorBleed)
            {
                algorithmMaterial.SetFloat("giPower", ColorBleedPower);
                if (Quality == 2 && ColorBleedQuality == 4)
                    algorithmMaterial.SetInt("giQuality", 2);
                else
                    algorithmMaterial.SetInt("giQuality", ColorBleedQuality);
            }

            //int quarterRadiusSamplesCount = DetailQuality == DetailQualityType.High ? 4 : 2;
            //int halfRadiusSamplesCount = DetailQuality == DetailQualityType.High ? 8 : 4;
            int quarterRadiusSamplesOffset = DetailQuality == DetailQualityType.High ? ((64 + 32 + 16 + 8) / 2) : ((64 + 32 + 16 + 8 + 4) / 2);
            int halfRadiusSamplesOffset = DetailQuality == DetailQualityType.High ? ((64 + 32 + 16) / 2) : ((64 + 32 + 16 + 8) / 2);

            SetVectorArrayNoBuffer("samples", algorithmMaterial, packedSamples);

            if (EnableTemporalFiltering)
            {
                //algorithmMaterial.SetInt("quarterRadiusSamplesCount", 2);
                algorithmMaterial.SetInt("quarterRadiusSamplesOffset", 18);
                //algorithmMaterial.SetInt("halfRadiusSamplesCount", 2);
                algorithmMaterial.SetInt("halfRadiusSamplesOffset", 18);
            } else
            {
                //algorithmMaterial.SetInt("quarterRadiusSamplesCount", quarterRadiusSamplesCount);
                algorithmMaterial.SetInt("quarterRadiusSamplesOffset", quarterRadiusSamplesOffset);
                //algorithmMaterial.SetInt("halfRadiusSamplesCount", halfRadiusSamplesCount);
                algorithmMaterial.SetInt("halfRadiusSamplesOffset", halfRadiusSamplesOffset);
            }
            // If simple -> go black
            if (Mode == VAOEffectCommandBuffer.EffectMode.Simple)
            {
                VAOFinalPassMaterial.SetColor("colorTint", Color.black);
            }
            else
            {
                VAOFinalPassMaterial.SetColor("colorTint", ColorTint);
            }

            if (BlurMode == BlurModeType.Enhanced)
            {
                if (gaussian == null || gaussian.Length != EnhancedBlurSize || EnhancedBlurDeviation != lastDeviation)
                {
                    gaussian = GenerateGaussian(EnhancedBlurSize, EnhancedBlurDeviation, out gaussianWeight, false);
                    lastDeviation = EnhancedBlurDeviation;
                }

                VAOFinalPassMaterial.SetFloat("gaussWeight", gaussianWeight);

                SetVectorArray("gauss", VAOFinalPassMaterial, gaussian, ref gaussianBuffer, ref lastEnhancedBlurSize, true);
            }

            SetKeywords(VAOFinalPassMaterial, "WFORCE_VAO_COLORBLEED_OFF", "WFORCE_VAO_COLORBLEED_ON", Mode == EffectMode.ColorBleed);
            SetKeywords(BeforeReflectionsBlendMaterial, "WFORCE_VAO_COLORBLEED_OFF", "WFORCE_VAO_COLORBLEED_ON", Mode == EffectMode.ColorBleed);
        }

        private void prepareAoHistory(int screenTextureWidth, int screenTextureHeight)
        {

            int bufferCount = 2;

            if (Mode == EffectMode.ColorBleed)
            {
                bufferCount = 8;
            }

            if (myCamera.stereoEnabled && !isSPSR)
            {
                bufferCount *= 2;
            }

            if (aoHistory == null || aoHistory.Length < bufferCount)
            {
                releaseAoHistory();

                aoHistory = new RenderTexture[bufferCount];

                for (int i = 0; i < bufferCount; i++)
                    aoHistory[i] = new RenderTexture(screenTextureWidth, screenTextureHeight, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear);
            }

        }

        private void releaseAoHistory()
        {            
            if (aoHistory != null)
            {
                for (int i = 0; i < aoHistory.Length; i++)
                    if (aoHistory[i] != null) aoHistory[i].Release();

                aoHistory = null;
            }
            historyReady = false;
        }

        private Vector4[] GetCombinedSamples(Vector4[] a, Vector4[] b, Vector4[] c = null, Vector4[] d = null)
        {
            int cLength = c == null ? 0 : c.Length;
            int dLength = d == null ? 0 : d.Length;

            if (carefulCache != null && carefulCache.Length == (a.Length + b.Length + cLength + dLength)) return carefulCache;

            carefulCache = new Vector4[a.Length + b.Length + cLength + dLength];

            // Force samples buffer to update
            lastSamplesLength = 0;

            Array.Copy(a, 0, carefulCache, 0, a.Length);
            Array.Copy(b, 0, carefulCache, a.Length, b.Length);

            if (c != null)
                Array.Copy(c, 0, carefulCache, a.Length + b.Length, c.Length);

            if (c != null && d != null)
                Array.Copy(d, 0, carefulCache, a.Length + b.Length + c.Length, d.Length);

            return carefulCache;
        }

        private Vector4[] selectSampleSet(int samplesCount)
        {
            switch (samplesCount)
            {
                case 64: return samp64;
                case 32: return samp32;
                case 16: return samp16;
                case 8: return samp8;
                case 4: return samp4;
                case 2: return samp2;
            }

            return null;
        }

        private int getAdaptiveSamplesOffset(int samplesCount)
        {
            switch (samplesCount)
            {
                case 32: return 0;
                case 16: return 32;
                case 8: return 32 + 16;
                case 4: return 32 + 16 + 8;
                case 2: return 32 + 16 + 8 + 4;
                default: return 0;
            }
        }

        private void SetKeywords(Material material, string offState, string onState, bool state)
        {
            if (state)
            {
                material.DisableKeyword(offState);
                material.EnableKeyword(onState);
            }
            else
            {
                material.DisableKeyword(onState);
                material.EnableKeyword(offState);
            }
        }

        private void EnsureMaterials()
        {
            if (vaoMainShader == null) vaoMainShader = Shader.Find("Hidden/Wilberforce/VAOStandardShader");
            if (vaoMainColorbleedShader == null) vaoMainColorbleedShader = Shader.Find("Hidden/Wilberforce/VAOStandardColorbleedShader");
            if (raycastMainShader == null) raycastMainShader = Shader.Find("Hidden/Wilberforce/RaycastShader");
            if (raycastMainColorbleedShader == null) raycastMainColorbleedShader = Shader.Find("Hidden/Wilberforce/RaycastColorbleedShader");
            if (vaoFinalPassShader == null) vaoFinalPassShader = Shader.Find("Hidden/Wilberforce/VAOFinalPassShader");
            if (vaoBeforeReflectionsBlendShader == null) vaoBeforeReflectionsBlendShader = Shader.Find("Hidden/Wilberforce/VAOBeforeReflectionsBlendShader");

            if (!VAOMaterial && vaoMainShader.isSupported)
            {
                VAOMaterial = CreateMaterial(vaoMainShader);
            }

            if (!VAOColorbleedMaterial && vaoMainColorbleedShader.isSupported)
            {
                VAOColorbleedMaterial = CreateMaterial(vaoMainColorbleedShader);
            }

            if (!RaycastMaterial && raycastMainShader.isSupported)
            {
                RaycastMaterial = CreateMaterial(raycastMainShader);
            }

            if (!RaycastColorbleedMaterial && raycastMainColorbleedShader.isSupported)
            {
                RaycastColorbleedMaterial = CreateMaterial(raycastMainColorbleedShader);
            }

            if (!VAOFinalPassMaterial && vaoFinalPassShader.isSupported)
            {
                VAOFinalPassMaterial = CreateMaterial(vaoFinalPassShader);
            }

            if (!BeforeReflectionsBlendMaterial && vaoBeforeReflectionsBlendShader.isSupported)
            {
                BeforeReflectionsBlendMaterial = CreateMaterial(vaoBeforeReflectionsBlendShader);
            }

            if (!vaoMainShader.isSupported || !raycastMainShader.isSupported || !vaoFinalPassShader.isSupported || !vaoBeforeReflectionsBlendShader.isSupported)
            {
                ReportError("Could not create shader (Shader not supported).");
            }
        }

        private static Material CreateMaterial(Shader shader)
        {
            if (!shader) return null;

            Material m = new Material(shader);
            m.hideFlags = HideFlags.HideAndDontSave;

            return m;
        }

        private static void DestroyMaterial(Material mat)
        {
            if (mat)
            {
                DestroyImmediate(mat);
                mat = null;
            }
        }

        private void SetVectorArrayNoBuffer(string name, Material material, Vector4[] samples)
        {
#if UNITY_5_4_OR_NEWER
            material.SetVectorArray(name, samples);
#else
            for (int i = 0; i < samples.Length; ++i)
            {
                material.SetVector(name + i.ToString(), samples[i]);
            }
#endif
        }

        private void SetVectorArray(string name, Material Material, Vector4[] samples, ref Vector4[] samplesBuffer, ref int lastBufferLength, bool needsUpdate)
        {
#if UNITY_5_4_OR_NEWER

            if (needsUpdate || lastBufferLength != samples.Length)
            {
                Array.Copy(samples, samplesBuffer, samples.Length);
                lastBufferLength = samples.Length;
            }

            Material.SetVectorArray(name, samplesBuffer);
#else
                    for (int i = 0; i < samples.Length; ++i)
                    {
                        Material.SetVector(name + i.ToString(), samples[i]);
                    }
#endif
        }

        private Vector4[] samplesPacked = null;

        private void SetSampleSet(string name, Material material, Vector4[] samples)
        {
            if (samplesPacked == null)
            {
                // Pack Vec2's into Vec4's
                samplesPacked = new Vector4[samples.Length / 2];

                for (int i = 0; i < samples.Length / 2; i++)
                {
                    var v1 = samples[i * 2];
                    var v2 = samples[i * 2 + 1];
                    samplesPacked[i] = new Vector4(v1.x, v1.y, v2.x, v2.y);
                }
            }

            SetVectorArray(name, material, samplesPacked, ref samplesLarge, ref lastSamplesLength, false);
        }

        #endregion

        #region VAO Data Utilities

        private Vector4[] GetAdaptiveSamples()
        {
            if (adaptiveSamples == null) adaptiveSamples = GenerateAdaptiveSamples();
            return adaptiveSamples;
        }

        private Vector4[] GenerateAdaptiveSamples()
        {
            Vector4[] result = new Vector4[62];

            Array.Copy(samp32, 0, result, 0, 32);
            Array.Copy(samp16, 0, result, 32, 16);
            Array.Copy(samp8, 0, result, 48, 8);
            Array.Copy(samp4, 0, result, 56, 4);
            Array.Copy(samp2, 0, result, 60, 2);

            return result;
        }

        private void EnsureNoiseTexture()
        {
            if (noiseTexture == null)
            {
                noiseTexture = new Texture2D(3, 3, TextureFormat.RGFloat, false, true);
                noiseTexture.SetPixels(noiseSamples);
                noiseTexture.filterMode = FilterMode.Point;
                noiseTexture.wrapMode = TextureWrapMode.Repeat;
                noiseTexture.Apply();
            }
        }

        private void EnsureTemporalSamplesTexture()
        {
            if (temporalSamplesTexture == null)
            {
                temporalSamplesTexture = new Texture2D(temporalSamplesPacked.Length, 1, TextureFormat.RGBAFloat, false, true);
                temporalSamplesTexture.SetPixels(temporalSamplesPacked);
                temporalSamplesTexture.filterMode = FilterMode.Point;
                temporalSamplesTexture.wrapMode = TextureWrapMode.Clamp;
                temporalSamplesTexture.Apply();
            }
        }

        private static Vector4[] GenerateGaussian(int size, float d, out float weight, bool normalize = true)
        {
            Vector4[] result = new Vector4[size];
            float norm = 0.0f;

            double twodd = 2.0 * d * d;
            double sqrt2ddpi = Math.Sqrt(twodd * Math.PI);

            float phase = (1.0f / (size + 1));
            for (int i = 0; i < size; i++)
            {
                float u = i / (float)(size + 1);
                u += phase;
                u *= 6.0f;
                float uminus3 = (u - 3.0f);

                float temp = -(float)(-(Math.Exp(-(uminus3 * uminus3) / twodd)) / sqrt2ddpi);

                result[i].x = temp;
                norm += temp;
            }

            if (normalize)
            {
                for (int i = 0; i < size; i++)
                {
                    result[i].x /= norm;
                }
            }

            weight = norm;

            return result;
        }

        #endregion

        #region VAO Implementation Utilities

        private float GetDepthForScreenSize(Camera camera, float sizeOnScreen)
        {
            return -(Radius * camera.projectionMatrix.m11) / sizeOnScreen;
        }

        public bool ShouldUseHierarchicalBuffer()
        {
            if (myCamera == null) return false;

            Vector2 texelSize = new Vector2(1.0f / myCamera.pixelWidth, 1.0f / myCamera.pixelHeight);

            float quarterResDistance = GetDepthForScreenSize(myCamera, Mathf.Max(texelSize.x, texelSize.y) * HierarchicalBufferPixelsPerLevel * 2.0f);
            quarterResDistance /= -myCamera.farClipPlane;

            return quarterResDistance > 0.1f;
        }

        public bool HierarchicalBufferEnabled
        {
            get
            {
                if (HierarchicalBufferState == HierarchicalBufferStateType.On) return true;

                if (HierarchicalBufferState == HierarchicalBufferStateType.Auto)
                {
                    return ShouldUseHierarchicalBuffer();
                }

                return false;
            }
        }

        public bool ShouldUseGBuffer()
        {
            if (myCamera == null) return UseGBuffer;

            if (myCamera.actualRenderingPath != RenderingPath.DeferredShading) return false;

            if (VaoCameraEvent != VAOCameraEventType.BeforeImageEffectsOpaque) return true;

            return UseGBuffer;
        }

        protected void EnsureVAOVersion()
        {
            if (CommandBufferEnabled && (this is VAOEffectCommandBuffer) && !(this is VAOEffect)) return;
            if (!CommandBufferEnabled && (this is VAOEffect)) return;

            var allComponents = GetComponents<Component>();
            var parameters = GetParameters();

            int oldComponentIndex = -1;
            Component newComponent = null;

            for (int i = 0; i < allComponents.Length; i++)
            {
                if (CommandBufferEnabled && (allComponents[i] == this))
                {

                    var oldGameObject = gameObject;
                    DestroyImmediate(this);
                    newComponent = oldGameObject.AddComponent<VAOEffectCommandBuffer>();
                    (newComponent as VAOEffectCommandBuffer).SetParameters(parameters);
                    oldComponentIndex = i;
                    break;
                }

                if (!CommandBufferEnabled && ((allComponents[i] == this)))
                {
                    var oldGameObject = gameObject;
                    TeardownCommandBuffer();
                    DestroyImmediate(this);
                    newComponent = oldGameObject.AddComponent<VAOEffect>();
                    (newComponent as VAOEffect).SetParameters(parameters);
                    oldComponentIndex = i;
                    break;
                }
            }

            if (oldComponentIndex >= 0 && newComponent != null)
            {
#if UNITY_EDITOR
                allComponents = newComponent.gameObject.GetComponents<Component>();
                int currentIndex = 0;

                for (int i = 0; i < allComponents.Length; i++)
                {
                    if (allComponents[i] == newComponent)
                    {
                        currentIndex = i;
                        break;
                    }
                }

                for (int i = 0; i < currentIndex - oldComponentIndex; i++)
                {
                    UnityEditorInternal.ComponentUtility.MoveComponentUp(newComponent);
                }
#endif
            }
        }

        private bool CheckSettingsChanges()
        {
            bool settingsDirty = false;

            if (GetCameraEvent(VaoCameraEvent) != lastCameraEvent)
            {
                TeardownCommandBuffer();
                settingsDirty = true;
            }

            if (((int)Algorithm) != lastAlgorithm)
            {
                if (lastAlgorithm != 0)
                {
                    if (Algorithm == AlgorithmType.RaycastAO)
                    {
                        Radius *= 0.5f;
                    }
                    else
                    {
                        Radius *= 2.0f;
                    }
                }

                lastAlgorithm = (int)Algorithm;
                settingsDirty = true;
            }

            if (EnableTemporalFiltering != lastEnableTemporalFiltering)
            {
                lastEnableTemporalFiltering = EnableTemporalFiltering;
                TeardownCommandBuffer();
                settingsDirty = true;
                historyReady = false;
            }

            if (OverrideCameraEvent != lastOverrideCameraEvent)
            {
                lastOverrideCameraEvent = OverrideCameraEvent;
                settingsDirty = true;
            }

            if (OverrideWidth != lastOverrideWidth)
            {
                lastOverrideWidth = OverrideWidth;
                settingsDirty = true;
            }

            if (OverrideHeight != lastOverrideHeight)
            {
                lastOverrideHeight = OverrideHeight;
                settingsDirty = true;
            }

            if (AdaptiveType != lastAdaptiveType)
            {
                lastAdaptiveType = AdaptiveType;
                carefulCache = null;
                samplesPacked = null;
            }

            if (Quality != lastQuality)
            {
                lastQuality = Quality;
                carefulCache = null;
                samplesPacked = null;
            }

            if (Downsampling != lastDownsampling)
            {
                lastDownsampling = Downsampling;
                TeardownCommandBuffer();
                settingsDirty = true;
            }

            if (CullingPrepassMode != lastcullingPrepassType)
            {
                lastcullingPrepassType = CullingPrepassMode;
                settingsDirty = true;
                carefulCache = null;
                samplesPacked = null;
            }

            if (CullingPrepassDownsamplingFactor != lastCullingPrepassDownsamplingFactor)
            {
                lastCullingPrepassDownsamplingFactor = CullingPrepassDownsamplingFactor;
                settingsDirty = true;
            }

            if (BlurMode != lastBlurMode)
            {
                lastBlurMode = BlurMode;
                settingsDirty = true;
            }

            if (Mode != lastMode)
            {
                lastMode = Mode;
                settingsDirty = true;
            }

            if (UseGBuffer != lastUseGBuffer)
            {
                lastUseGBuffer = UseGBuffer;
                settingsDirty = true;
            }

            if (OutputAOOnly != lastOutputAOOnly)
            {
                lastOutputAOOnly = OutputAOOnly;
                settingsDirty = true;
            }

            isHDR = isCameraHDR(myCamera);
            if (isHDR != lastIsHDR)
            {
                lastIsHDR = isHDR;
                settingsDirty = true;
            }

#if UNITY_EDITOR
            isSPSR = isCameraSPSR(myCamera);
            if (isSPSR != lastIsSPSR)
            {
                lastIsSPSR = isSPSR;
                settingsDirty = true;
            }

            isMPSR = myCamera.stereoEnabled && !isSPSR;
            if (isMPSR != lastIsMPSR)
            {
                lastIsMPSR = isMPSR;
                TeardownCommandBuffer();
                releaseAoHistory();
                settingsDirty = true;
                historyReady = false;
            }
#endif


            if (lastIntermediateScreenTextureFormat != IntermediateScreenTextureFormat)
            {
                lastIntermediateScreenTextureFormat = IntermediateScreenTextureFormat;
                settingsDirty = true;
            }

            if (lastCmdBufferEnhancedBlurSize != EnhancedBlurSize)
            {
                lastCmdBufferEnhancedBlurSize = EnhancedBlurSize;
                settingsDirty = true;
            }

            if (lastHierarchicalBufferEnabled != HierarchicalBufferEnabled)
            {
                lastHierarchicalBufferEnabled = HierarchicalBufferEnabled;
                settingsDirty = true;
            }

            if (lastBlurQuality != BlurQuality)
            {
                lastBlurQuality = BlurQuality;
                settingsDirty = true;
            }

            float DetailAmount = Algorithm == AlgorithmType.StandardVAO ? DetailAmountVAO : DetailAmountRaycast;

            int MainPass = (int)GetMainPass(Algorithm, DetailAmount > 0.5f, DetailAmount > 0.0f);

            if (lastMainPass != MainPass)
            {
                lastMainPass = MainPass;
                settingsDirty = true;
                carefulCache = null;
                samplesPacked = null;
            }

            return settingsDirty;
        }

        private RenderTextureFormat GetRenderTextureFormat(ScreenTextureFormat format, bool isHDR)
        {
            switch (format)
            {
                case ScreenTextureFormat.Default:
                    return RenderTextureFormat.Default;
                case ScreenTextureFormat.DefaultHDR:
                    return RenderTextureFormat.DefaultHDR;
                case ScreenTextureFormat.ARGB32:
                    return RenderTextureFormat.ARGB32;
                case ScreenTextureFormat.ARGBFloat:
                    return RenderTextureFormat.ARGBFloat;
                case ScreenTextureFormat.ARGBHalf:
                    return RenderTextureFormat.ARGBHalf;
                default:
                    return isHDR ? RenderTextureFormat.DefaultHDR : RenderTextureFormat.Default;
            }
        }

        #endregion

        #region Unity Utilities

        private void ReportError(string error)
        {
            if (Debug.isDebugBuild) Debug.LogError("VAO Effect Error: " + error);
        }

        private void ReportWarning(string error)
        {
            if (Debug.isDebugBuild) Debug.LogWarning("VAO Effect Warning: " + error);
        }

        private bool isCameraSPSR(Camera camera)
        {
            if (camera == null) return false;

#if UNITY_5_5_OR_NEWER
            if (camera.stereoEnabled)
            {
#if UNITY_2017_2_OR_NEWER

                return (UnityEngine.XR.XRSettings.eyeTextureDesc.vrUsage == VRTextureUsage.TwoEyes);
#else

#if !UNITY_WEBGL
#if UNITY_EDITOR
                if (camera.stereoEnabled && PlayerSettings.stereoRenderingPath == StereoRenderingPath.SinglePass)
                    return true;
#endif
#endif

#endif
            }
#endif

            return false;
        }

        private bool isCameraHDR(Camera camera)
        {

#if UNITY_5_6_OR_NEWER
            if (camera != null) return camera.allowHDR;
#else
            if (camera != null) return camera.hdr;
#endif
            return false;
        }

        private bool MustForceFlip(Camera camera)
        {
#if UNITY_5_6_OR_NEWER
            return false;
#else
            if (myCamera.stereoEnabled)
            {
                return false;
            }
            if (!CommandBufferEnabled) return false;
            if (camera.actualRenderingPath != RenderingPath.DeferredShading && camera.actualRenderingPath != RenderingPath.DeferredLighting) return true;
            return false;
#endif
        }

        protected List<KeyValuePair<FieldInfo, object>> GetParameters()
        {
            var result = new List<KeyValuePair<FieldInfo, object>>();

            var fields = this.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public);

            foreach (var field in fields)
            {
                result.Add(new KeyValuePair<FieldInfo, object>(field, field.GetValue(this)));
            }

            return result;
        }

        protected void SetParameters(List<KeyValuePair<FieldInfo, object>> parameters)
        {
            foreach (var parameter in parameters)
            {
                parameter.Key.SetValue(this, parameter.Value);
            }
        }

        #endregion

        #region  Data

        //private static float[] adaptiveLengths = new float[16] { 32, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, /*2, */4 };
        //private static float[] adaptiveStarts = new float[16] { 32, 48, 48, 56, 56, 56, 56, 60, 60, 60, 60, 60, 60, 60, 60, /*60,*/ 60 };

        private static Color[] noiseSamples = new Color[9]{
                new Color(1.0f, 0.0f, 0.0f),
                new Color(-0.939692f, 0.342022f, 0.0f),
                new Color(0.173644f, -0.984808f, 0.0f),
                new Color(0.173649f, 0.984808f, 0.0f),
                new Color(-0.500003f, -0.866024f, 0.0f),
                new Color(0.766045f, 0.642787f, 0.0f),
                new Color(-0.939694f, -0.342017f, 0.0f),
                new Color(0.766042f, -0.642791f, 0.0f),
                new Color(-0.499999f, 0.866026f, 0.0f)};
        private static Vector4[] samp2 = new Vector4[2] {
                new Vector4(0.4392292f,  0.0127914f, 0.898284f),
                new Vector4(-0.894406f,  -0.162116f, 0.41684f)};
        private static Vector4[] samp4 = new Vector4[4] {
                new Vector4(-0.07984404f,  -0.2016976f, 0.976188f),
                new Vector4(0.4685118f,  -0.8404996f, 0.272135f),
                new Vector4(-0.793633f,  0.293059f, 0.533164f),
                new Vector4(0.2998218f,  0.4641494f, 0.83347f)};
        private static Vector4[] samp8 = new Vector4[8] {
                new Vector4(-0.4999112f,  -0.571184f, 0.651028f),
                new Vector4(0.2267525f,  -0.668142f, 0.708639f),
                new Vector4(0.0657284f,  -0.123769f, 0.990132f),
                new Vector4(0.9259827f,  -0.2030669f, 0.318307f),
                new Vector4(-0.9850165f,  0.1247843f, 0.119042f),
                new Vector4(-0.2988613f,  0.2567392f, 0.919112f),
                new Vector4(0.4734727f,  0.2830991f, 0.834073f),
                new Vector4(0.1319883f,  0.9544416f, 0.267621f)};
        private static Vector4[] samp16 = new Vector4[16] {
                new Vector4(-0.6870962f,  -0.7179669f, 0.111458f),
                new Vector4(-0.2574025f,  -0.6144419f, 0.745791f),
                new Vector4(-0.408366f,  -0.162244f, 0.898284f),
                new Vector4(-0.07098053f,  0.02052395f, 0.997267f),
                new Vector4(0.2019972f,  -0.760972f, 0.616538f),
                new Vector4(0.706282f,  -0.6368136f, 0.309248f),
                new Vector4(0.169605f,  -0.2892981f, 0.942094f),
                new Vector4(0.7644456f,  -0.05826119f, 0.64205f),
                new Vector4(-0.745912f,  0.0501786f, 0.664152f),
                new Vector4(-0.7588732f,  0.4313389f, 0.487911f),
                new Vector4(-0.3806622f,  0.3446409f, 0.85809f),
                new Vector4(-0.1296651f,  0.8794711f, 0.45795f),
                new Vector4(0.1557318f,  0.137468f, 0.978187f),
                new Vector4(0.5990864f,  0.2485375f, 0.761133f),
                new Vector4(0.1727637f,  0.5753375f, 0.799462f),
                new Vector4(0.5883294f,  0.7348878f, 0.337355f)};
        private static Vector4[] samp32 = new Vector4[32] {
                new Vector4(-0.626056f,  -0.7776781f, 0.0571977f),
                new Vector4(-0.1335098f,  -0.9164876f, 0.377127f),
                new Vector4(-0.2668636f,  -0.5663173f, 0.779787f),
                new Vector4(-0.5712572f,  -0.4639561f, 0.67706f),
                new Vector4(-0.6571807f,  -0.2969118f, 0.692789f),
                new Vector4(-0.8896923f,  -0.1314662f, 0.437223f),
                new Vector4(-0.5037534f,  -0.03057539f, 0.863306f),
                new Vector4(-0.1773856f,  -0.2664998f, 0.947371f),
                new Vector4(-0.02786797f,  -0.02453661f, 0.99931f),
                new Vector4(0.173095f,  -0.964425f, 0.199805f),
                new Vector4(0.280491f,  -0.716259f, 0.638982f),
                new Vector4(0.7610048f,  -0.4987299f, 0.414898f),
                new Vector4(0.135136f,  -0.388973f, 0.911284f),
                new Vector4(0.4836829f,  -0.4782286f, 0.73304f),
                new Vector4(0.1905736f,  -0.1039435f, 0.976154f),
                new Vector4(0.4855643f,  0.01388972f, 0.87409f),
                new Vector4(0.5684234f,  -0.2864941f, 0.771243f),
                new Vector4(0.8165832f,  0.01384446f, 0.577062f),
                new Vector4(-0.9814694f,  0.18555f, 0.0478435f),
                new Vector4(-0.5357604f,  0.3316899f, 0.776494f),
                new Vector4(-0.1238877f,  0.03315933f, 0.991742f),
                new Vector4(-0.1610546f,  0.3801286f, 0.910804f),
                new Vector4(-0.5923722f,  0.628729f, 0.503781f),
                new Vector4(-0.05504921f,  0.5483891f, 0.834409f),
                new Vector4(-0.3805041f,  0.8377199f, 0.391717f),
                new Vector4(-0.101651f,  0.9530866f, 0.285119f),
                new Vector4(0.1613653f,  0.2561041f, 0.953085f),
                new Vector4(0.4533991f,  0.2896196f, 0.842941f),
                new Vector4(0.6665574f,  0.4639243f, 0.583503f),
                new Vector4(0.8873722f,  0.4278904f, 0.1717f),
                new Vector4(0.2869751f,  0.732805f, 0.616962f),
                new Vector4(0.4188429f,  0.7185978f, 0.555147f)};
        private static Vector4[] samp64 = new Vector4[64] {
                new Vector4(-0.6700248f,  -0.6370129f, 0.381157f),
                new Vector4(-0.7385408f,  -0.6073685f, 0.292679f),
                new Vector4(-0.4108568f,  -0.8852778f, 0.2179f),
                new Vector4(-0.3058583f,  -0.8047022f, 0.508828f),
                new Vector4(0.01087609f,  -0.7610992f, 0.648545f),
                new Vector4(-0.3629634f,  -0.5480431f, 0.753595f),
                new Vector4(-0.1480379f,  -0.6927805f, 0.70579f),
                new Vector4(-0.9533184f,  -0.276674f, 0.12098f),
                new Vector4(-0.6387863f,  -0.3999016f, 0.65729f),
                new Vector4(-0.891588f,  -0.115146f, 0.437964f),
                new Vector4(-0.775663f,  0.0194654f, 0.630848f),
                new Vector4(-0.5360528f,  -0.1828935f, 0.824134f),
                new Vector4(-0.513927f,  -0.000130296f, 0.857834f),
                new Vector4(-0.4368436f,  -0.2831443f, 0.853813f),
                new Vector4(-0.1794069f,  -0.4226944f, 0.888337f),
                new Vector4(-0.00183062f,  -0.4371257f, 0.899398f),
                new Vector4(-0.2598701f,  -0.1719497f, 0.950211f),
                new Vector4(-0.08650014f,  -0.004176182f, 0.996243f),
                new Vector4(0.006921067f,  -0.001478712f, 0.999975f),
                new Vector4(0.05654667f,  -0.9351676f, 0.349662f),
                new Vector4(0.1168661f,  -0.754741f, 0.64553f),
                new Vector4(0.3534952f,  -0.7472929f, 0.562667f),
                new Vector4(0.1635596f,  -0.5863093f, 0.793404f),
                new Vector4(0.5910167f,  -0.786864f, 0.177609f),
                new Vector4(0.5820105f,  -0.5659724f, 0.5839f),
                new Vector4(0.7254612f,  -0.5323696f, 0.436221f),
                new Vector4(0.4016336f,  -0.4329237f, 0.807012f),
                new Vector4(0.5287027f,  -0.4064075f, 0.745188f),
                new Vector4(0.314015f,  -0.2375291f, 0.919225f),
                new Vector4(0.02922117f,  -0.2097672f, 0.977315f),
                new Vector4(0.4201531f,  -0.1445212f, 0.895871f),
                new Vector4(0.2821195f,  -0.01079273f, 0.959319f),
                new Vector4(0.7152653f,  -0.1972963f, 0.670425f),
                new Vector4(0.8167331f,  -0.1217311f, 0.564029f),
                new Vector4(0.8517836f,  0.01290532f, 0.523735f),
                new Vector4(-0.657816f,  0.134013f, 0.74116f),
                new Vector4(-0.851676f,  0.321285f, 0.414033f),
                new Vector4(-0.603183f,  0.361627f, 0.710912f),
                new Vector4(-0.6607267f,  0.5282444f, 0.533289f),
                new Vector4(-0.323619f,  0.182656f, 0.92839f),
                new Vector4(-0.2080927f,  0.1494067f, 0.966631f),
                new Vector4(-0.4205947f,  0.4184987f, 0.804959f),
                new Vector4(-0.06831062f,  0.3712724f, 0.926008f),
                new Vector4(-0.165943f,  0.5029928f, 0.84821f),
                new Vector4(-0.6137413f,  0.7001954f, 0.364758f),
                new Vector4(-0.3009551f,  0.6550035f, 0.693107f),
                new Vector4(-0.1356791f,  0.6460465f, 0.751143f),
                new Vector4(-0.3677429f,  0.7920387f, 0.487278f),
                new Vector4(-0.08688695f,  0.9677781f, 0.236338f),
                new Vector4(0.07250954f,  0.1327261f, 0.988497f),
                new Vector4(0.5244588f,  0.05565827f, 0.849615f),
                new Vector4(0.2498424f,  0.3364912f, 0.907938f),
                new Vector4(0.2608168f,  0.5340923f, 0.804189f),
                new Vector4(0.3888291f,  0.3207975f, 0.863655f),
                new Vector4(0.6413552f,  0.1619097f, 0.749966f),
                new Vector4(0.8523082f,  0.2647078f, 0.451111f),
                new Vector4(0.5591328f,  0.3038472f, 0.771393f),
                new Vector4(0.9147445f,  0.3917669f, 0.0987938f),
                new Vector4(0.08110893f,  0.7317293f, 0.676752f),
                new Vector4(0.3154335f,  0.7388063f, 0.59554f),
                new Vector4(0.1677455f,  0.9625717f, 0.212877f),
                new Vector4(0.3015989f,  0.9509261f, 0.069128f),
                new Vector4(0.5600207f,  0.5649592f, 0.605969f),
                new Vector4(0.6455291f,  0.7387806f, 0.193637f)};

        private static Vector4[] packedSamples = new Vector4[63] {
                new Vector4(-0.6700248f, -0.6370129f, -0.7385408f, -0.6073685f),
                new Vector4(-0.4108568f, -0.8852778f, -0.3058583f, -0.8047022f),
                new Vector4(0.01087609f, -0.7610992f, -0.3629634f, -0.5480431f),
                new Vector4(-0.1480379f, -0.6927805f, -0.9533184f, -0.276674f),
                new Vector4(-0.6387863f, -0.3999016f, -0.891588f, -0.115146f),
                new Vector4(-0.775663f, 0.0194654f, -0.5360528f, -0.1828935f),
                new Vector4(-0.513927f, -0.000130296f, -0.4368436f, -0.2831443f),
                new Vector4(-0.1794069f, -0.4226944f, -0.00183062f, -0.4371257f),
                new Vector4(-0.2598701f, -0.1719497f, -0.08650014f, -0.004176182f),
                new Vector4(0.006921067f, -0.001478712f, 0.05654667f, -0.9351676f),
                new Vector4(0.1168661f, -0.754741f, 0.3534952f, -0.7472929f),
                new Vector4(0.1635596f, -0.5863093f, 0.5910167f, -0.786864f),
                new Vector4(0.5820105f, -0.5659724f, 0.7254612f, -0.5323696f),
                new Vector4(0.4016336f, -0.4329237f, 0.5287027f, -0.4064075f),
                new Vector4(0.314015f, -0.2375291f, 0.02922117f, -0.2097672f),
                new Vector4(0.4201531f, -0.1445212f, 0.2821195f, -0.01079273f),
                new Vector4(0.7152653f, -0.1972963f, 0.8167331f, -0.1217311f),
                new Vector4(0.8517836f, 0.01290532f, -0.657816f, 0.134013f),
                new Vector4(-0.851676f, 0.321285f, -0.603183f, 0.361627f),
                new Vector4(-0.6607267f, 0.5282444f, -0.323619f, 0.182656f),
                new Vector4(-0.2080927f, 0.1494067f, -0.4205947f, 0.4184987f),
                new Vector4(-0.06831062f, 0.3712724f, -0.165943f, 0.5029928f),
                new Vector4(-0.6137413f, 0.7001954f, -0.3009551f, 0.6550035f),
                new Vector4(-0.1356791f, 0.6460465f, -0.3677429f, 0.7920387f),
                new Vector4(-0.08688695f, 0.9677781f, 0.07250954f, 0.1327261f),
                new Vector4(0.5244588f, 0.05565827f, 0.2498424f, 0.3364912f),
                new Vector4(0.2608168f, 0.5340923f, 0.3888291f, 0.3207975f),
                new Vector4(0.6413552f, 0.1619097f, 0.8523082f, 0.2647078f),
                new Vector4(0.5591328f, 0.3038472f, 0.9147445f, 0.3917669f),
                new Vector4(0.08110893f, 0.7317293f, 0.3154335f, 0.7388063f),
                new Vector4(0.1677455f, 0.9625717f, 0.3015989f, 0.9509261f),
                new Vector4(0.5600207f, 0.5649592f, 0.6455291f, 0.7387806f),
                new Vector4(-0.626056f, -0.7776781f, -0.1335098f, -0.9164876f),
                new Vector4(-0.2668636f, -0.5663173f, -0.5712572f, -0.4639561f),
                new Vector4(-0.6571807f, -0.2969118f, -0.8896923f, -0.1314662f),
                new Vector4(-0.5037534f, -0.03057539f, -0.1773856f, -0.2664998f),
                new Vector4(-0.02786797f, -0.02453661f, 0.173095f, -0.964425f),
                new Vector4(0.280491f, -0.716259f, 0.7610048f, -0.4987299f),
                new Vector4(0.135136f, -0.388973f, 0.4836829f, -0.4782286f),
                new Vector4(0.1905736f, -0.1039435f, 0.4855643f, 0.01388972f),
                new Vector4(0.5684234f, -0.2864941f, 0.8165832f, 0.01384446f),
                new Vector4(-0.9814694f, 0.18555f, -0.5357604f, 0.3316899f),
                new Vector4(-0.1238877f, 0.03315933f, -0.1610546f, 0.3801286f),
                new Vector4(-0.5923722f, 0.628729f, -0.05504921f, 0.5483891f),
                new Vector4(-0.3805041f, 0.8377199f, -0.101651f, 0.9530866f),
                new Vector4(0.1613653f, 0.2561041f, 0.4533991f, 0.2896196f),
                new Vector4(0.6665574f, 0.4639243f, 0.8873722f, 0.4278904f),
                new Vector4(0.2869751f, 0.732805f, 0.4188429f, 0.7185978f),
                new Vector4(-0.6870962f, -0.7179669f, -0.2574025f, -0.6144419f),
                new Vector4(-0.408366f, -0.162244f, -0.07098053f, 0.02052395f),
                new Vector4(0.2019972f, -0.760972f, 0.706282f, -0.6368136f),
                new Vector4(0.169605f, -0.2892981f, 0.7644456f, -0.05826119f),
                new Vector4(-0.745912f, 0.0501786f, -0.7588732f, 0.4313389f),
                new Vector4(-0.3806622f, 0.3446409f, -0.1296651f, 0.8794711f),
                new Vector4(0.1557318f, 0.137468f, 0.5990864f, 0.2485375f),
                new Vector4(0.1727637f, 0.5753375f, 0.5883294f, 0.7348878f),
                new Vector4(-0.4999112f, -0.571184f, 0.2267525f, -0.668142f),
                new Vector4(0.0657284f, -0.123769f, 0.9259827f, -0.2030669f),
                new Vector4(-0.9850165f, 0.1247843f, -0.2988613f, 0.2567392f),
                new Vector4(0.4734727f, 0.2830991f, 0.1319883f, 0.9544416f),

                //new Vector4(-0.07984404f, -0.2016976f, 0.4685118f, -0.8404996f),
                //new Vector4(-0.793633f, 0.293059f, 0.2998218f, 0.4641494f),
                new Vector4(-0.153437f, 0.388008f, -0.0705125f, 0.677141f),
                new Vector4(-0.100007f, 0.0368103f, -0.344102f, 0.817498f),
                new Vector4(0.4392292f, 0.0127914f, -0.894406f, -0.162116f)
                };

        private static Color[] temporalSamplesPacked = new Color[558]
        {
            new Color(-0.573172f, -0.532465f, 0.587828f, 0.380267f),
new Color(-0.857373f, -0.448367f, 0.322927f, -0.208062f),
new Color(-0.770474f, 0.378665f, 0.367633f, -0.868896f),
new Color(-0.0738539f, 0.628146f, -0.383834f, -0.922466f),
new Color(-0.139787f, -0.145552f, 0.825291f, 0.495564f),
new Color(-0.894489f, -0.0617844f, 0.600867f, 0.0580676f),
new Color(-0.693786f, 0.696058f, 0.495962f, -0.566117f),
new Color(-0.272272f, 0.830458f, -0.100632f, -0.788655f),
new Color(-0.629072f, -0.276978f, 0.0649682f, -0.439588f),
new Color(-0.340094f, -0.656093f, 0.934086f, -0.248247f),
new Color(-0.67346f, 0.11431f, 0.759222f, -0.60907f),
new Color(0.562448f, 0.821412f, 0.117007f, -0.9522f),
new Color(0.23493f, 0.591106f, 0.0140933f, 0.243143f),
new Color(-0.426217f, 0.465168f, -0.0281225f, 0.950881f),
new Color(-0.37139f, 0.0179654f, 0.879943f, 0.09586f),
new Color(0.260057f, 0.837749f, 0.172604f, -0.681421f),
new Color(-0.209746f, 0.323099f, -0.360208f, -0.373921f),
new Color(0.620504f, -0.268533f, 0.248378f, 0.0877057f),
new Color(-0.111673f, -0.805692f, -0.0921714f, 0.198186f),
new Color(-0.41279f, -0.00623369f, 0.386095f, 0.741015f),
new Color(0.243681f, -0.902202f, -0.966973f, 0.0506905f),
new Color(-0.451944f, -0.391673f, 0.470578f, 0.213208f),
new Color(0.730174f, -0.629536f, -0.505434f, 0.453231f),
new Color(-0.847213f, -0.420726f, 0.88622f, -0.135896f),
new Color(0.472295f, -0.378033f, -0.433324f, 0.899029f),
new Color(-0.512198f, -0.84487f, 0.33882f, -0.0621014f),
new Color(-0.00554973f, -0.425101f, -0.20188f, 0.619922f),
new Color(-0.0648268f, 0.985998f, 0.497291f, -0.855726f),
new Color(0.923471f, 0.129255f, -0.353302f, -0.681945f),
new Color(0.547634f, 0.828596f, 0.452132f, -0.217132f),
new Color(0.601841f, 0.335957f, -0.662233f, -0.374095f),
new Color(-0.830721f, 0.36586f, 0.134286f, -0.458476f),
new Color(0.666723f, -0.493933f, -0.357254f, 0.138589f),
new Color(-0.0696585f, 0.765169f, -0.164765f, -0.54802f),
new Color(-0.342f, 0.43318f, -0.269501f, -0.381071f),
new Color(0.112766f, 0.241213f, 0.253631f, 0.121867f),
new Color(0.0874627f, -0.98784f, -0.248589f, 0.827489f),
new Color(-0.894474f, -0.173916f, 0.241316f, 0.860883f),
new Color(-0.301417f, -0.940516f, -0.133819f, 0.458456f),
new Color(-0.59478f, 0.283413f, 0.602431f, 0.573521f),
new Color(0.887774f, -0.383196f, 0.294239f, 0.284313f),
new Color(-0.416595f, 0.721009f, 0.542035f, 0.0515079f),
new Color(0.42716f, -0.685994f, -0.181728f, 0.0109197f),
new Color(0.70661f, 0.114861f, 0.292003f, -0.34892f),
new Color(-0.46086f, -0.174902f, -0.0058912f, -0.249559f),
new Color(-0.685888f, 0.674723f, 0.10816f, -0.640948f),
new Color(0.793963f, 0.589964f, -0.70433f, -0.201913f),
new Color(0.841996f, 0.399802f, -0.731786f, 0.0376775f),
new Color(0.745759f, -0.314864f, -0.211147f, -0.243536f),
new Color(-0.801454f, 0.523342f, 0.162043f, -0.186723f),
new Color(0.347945f, -0.52351f, 0.000780463f, 0.522231f),
new Color(0.109733f, 0.7145f, 0.40126f, 0.517453f),
new Color(-0.644107f, -0.539325f, 0.215646f, 0.44983f),
new Color(0.650355f, -0.164203f, 0.0814569f, 0.0663118f),
new Color(-0.390634f, -0.6522f, 0.137677f, -0.546363f),
new Color(-0.336191f, 0.216193f, 0.102813f, 0.386533f),
new Color(-0.860669f, -0.492747f, 0.179158f, -0.0335407f),
new Color(0.853953f, -0.114654f, 0.0102977f, 0.942333f),
new Color(-0.0296069f, -0.726607f, 0.169038f, -0.908303f),
new Color(-0.95366f, 0.0707179f, 0.411431f, 0.311704f),
new Color(-0.924065f, -0.222466f, -0.034844f, -0.193246f),
new Color(0.480424f, -0.0118935f, -0.190246f, 0.729656f),
new Color(-0.0892423f, -0.993831f, 0.460571f, -0.746162f),
new Color(-0.748474f, 0.277033f, 0.943545f, 0.224549f),
new Color(-0.672377f, -0.0867361f, -0.347078f, -0.0873827f),
new Color(0.684807f, -0.311489f, -0.608949f, 0.786797f),
new Color(-0.375845f, -0.902154f, 0.648746f, -0.629388f),
new Color(-0.789415f, 0.493358f, 0.499097f, 0.750078f),
new Color(-0.532094f, -0.440568f, -0.255664f, -0.438146f),
new Color(-0.0621458f, 0.0616926f, -0.200159f, 0.386272f),
new Color(-0.579263f, -0.812408f, 0.406099f, -0.441059f),
new Color(-0.446954f, 0.493021f, 0.169537f, 0.672549f),
new Color(0.464456f, 0.875859f, -0.715145f, 0.607413f),
new Color(0.0761304f, -0.996828f, -0.46812f, -0.547798f),
new Color(0.826646f, 0.406941f, -0.984976f, -0.081476f),
new Color(-0.447215f, -0.778893f, -0.420773f, -0.305459f),
new Color(0.140043f, 0.983766f, -0.959085f, 0.182121f),
new Color(0.53557f, -0.62069f, -0.253695f, -0.281484f),
new Color(0.926393f, 0.0078907f, -0.916206f, -0.376655f),
new Color(-0.266703f, -0.945742f, 0.297287f, -0.0533695f),
new Color(-0.474479f, 0.873f, -0.65051f, 0.181042f),
new Color(0.298037f, -0.885616f, -0.331871f, 0.354257f),
new Color(0.841818f, -0.24546f, 0.704161f, -0.501873f),
new Color(-0.535979f, -0.0931597f, 0.440766f, 0.116458f),
new Color(0.309806f, 0.731478f, -0.322001f, 0.464757f),
new Color(0.56277f, -0.356101f, -0.295348f, 0.0361791f),
new Color(0.534566f, 0.410579f, 0.0669951f, 0.522813f),
new Color(0.151187f, -0.680472f, 0.0512714f, -0.0271371f),
new Color(-0.047559f, 0.764981f, -0.189256f, 0.537003f),
new Color(0.192415f, -0.422155f, -0.0467702f, 0.208916f),
new Color(-0.561881f, -0.642148f, 0.970178f, -0.226543f),
new Color(-0.234692f, 0.937908f, 0.298317f, 0.403625f),
new Color(-0.709625f, -0.501003f, 0.645027f, 0.695191f),
new Color(0.316946f, 0.872519f, 0.370107f, -0.200194f),
new Color(-0.287536f, -0.785107f, 0.719866f, -0.0923268f),
new Color(-0.352856f, 0.611746f, -0.0208867f, 0.344649f),
new Color(-0.818855f, 0.125091f, 0.647126f, 0.452989f),
new Color(0.0896137f, 0.827422f, -0.639705f, -0.283298f),
new Color(0.618983f, -0.740359f, 0.554837f, -0.491453f),
new Color(-0.0792019f, 0.529802f, 0.128125f, -0.30475f),
new Color(-0.631872f, 0.503359f, -0.357161f, 0.859488f),
new Color(0.564701f, 0.156425f, -0.0202714f, -0.383473f),
new Color(0.0477432f, -0.828224f, 0.309765f, -0.611821f),
new Color(-0.69558f, 0.0505533f, 0.13251f, 0.130565f),
new Color(-0.776545f, -0.170768f, -0.255819f, -0.625534f),
new Color(0.418516f, 0.474732f, -0.202337f, 0.139102f),
new Color(0.363534f, -0.727339f, 0.0113889f, -0.519204f),
new Color(-0.546336f, 0.262712f, -0.174712f, -0.052854f),
new Color(0.771837f, 0.587744f, -0.828806f, -0.0421628f),
new Color(-0.167485f, -0.735594f, -0.575383f, 0.384694f),
new Color(0.610375f, 0.54918f, -0.734435f, -0.65434f),
new Color(-0.361025f, -0.514422f, -0.414274f, 0.0875578f),
new Color(-0.103217f, 0.93529f, -0.765781f, -0.36018f),
new Color(0.428122f, -0.54917f, 0.506694f, -0.159087f),
new Color(0.87715f, -0.447626f, -0.138816f, -0.870934f),
new Color(-0.125746f, -0.522739f, 0.262786f, 0.584786f),
new Color(-0.879572f, 0.322952f, -0.473778f, 0.621974f),
new Color(0.309945f, -0.337078f, -0.163784f, 0.270479f),
new Color(0.8198f, -0.569273f, 0.459482f, -0.880552f),
new Color(-0.542116f, -0.24236f, 0.186253f, 0.280349f),
new Color(0.185817f, 0.819586f, -0.338316f, 0.727947f),
new Color(0.689812f, 0.0372716f, -0.232089f, -0.179794f),
new Color(0.79989f, 0.194191f, 0.464642f, 0.623217f),
new Color(-0.541547f, 0.0440183f, 0.205901f, -0.198658f),
new Color(-0.457855f, 0.735466f, -0.0775658f, 0.638211f),
new Color(0.662408f, 0.270384f, 0.274274f, 0.167354f),
new Color(-0.24411f, -0.604046f, -0.77127f, -0.344243f),
new Color(-0.148215f, -0.234011f, 0.445113f, -0.670569f),
new Color(0.550208f, -0.0543095f, -0.382437f, 0.212247f),
new Color(-0.103154f, 0.542619f, 0.459056f, 0.569972f),
new Color(-0.611457f, -0.759454f, -0.389729f, -0.278432f),
new Color(0.0056411f, 0.0150388f, 0.390242f, -0.307177f),
new Color(-0.924525f, 0.229696f, -0.402111f, 0.727562f),
new Color(0.400097f, 0.291763f, 0.886173f, 0.440402f),
new Color(-0.215465f, -0.801464f, -0.918487f, -0.391735f),
new Color(-0.0298064f, -0.477942f, 0.482047f, -0.827419f),
new Color(0.982405f, -0.0861904f, -0.174366f, 0.267324f),
new Color(-0.238034f, 0.685069f, 0.193675f, 0.612408f),
new Color(-0.748926f, -0.593039f, -0.267803f, -0.36226f),
new Color(-0.204202f, 0.0131948f, 0.232586f, -0.0593277f),
new Color(-0.772989f, 0.108249f, -0.54428f, 0.605853f),
new Color(0.476219f, 0.18123f, 0.684015f, 0.367319f),
new Color(0.00264335f, -0.964748f, -0.921891f, -0.204997f),
new Color(0.278303f, -0.899563f, 0.634281f, -0.758986f),
new Color(0.76097f, -0.0585868f, 0.00944293f, 0.186976f),
new Color(-0.412819f, 0.89344f, 0.227916f, 0.780267f),
new Color(-0.516866f, -0.617607f, -0.00528133f, -0.697443f),
new Color(-0.277937f, -0.122584f, 0.0459678f, -0.301251f),
new Color(-0.646057f, 0.184739f, -0.749253f, 0.575568f),
new Color(0.209369f, 0.369456f, 0.802619f, 0.154751f),
new Color(-0.165961f, -0.98532f, -0.959404f, -0.0538259f),
new Color(0.222801f, -0.746154f, 0.827663f, -0.535645f),
new Color(0.863065f, -0.244403f, -0.0109636f, 0.341112f),
new Color(-0.141011f, 0.880011f, 0.110449f, 0.989326f),
new Color(-0.391474f, -0.559122f, -0.621453f, -0.415902f),
new Color(-0.47695f, 0.0068202f, 0.180497f, -0.494202f),
new Color(-0.810704f, 0.423336f, -0.574984f, 0.3842f),
new Color(0.239663f, 0.1558f, 0.649634f, 0.231576f),
new Color(-0.386515f, -0.899851f, -0.694637f, -0.143852f),
new Color(-0.0376765f, -0.153728f, 0.385174f, -0.485069f),
new Color(0.603126f, -0.358054f, -0.317157f, 0.414322f),
new Color(0.0213674f, 0.82572f, 0.501144f, 0.760907f),
new Color(0.19517f, 0.958915f, 0.981605f, 0.0155536f),
new Color(-0.914374f, 0.357832f, -0.912644f, 0.06275f),
new Color(-0.680856f, -0.527844f, -0.157502f, -0.888457f),
new Color(-0.514128f, -0.359706f, -0.0373927f, -0.0637253f),
new Color(0.792485f, 0.549495f, 0.960471f, -0.192264f),
new Color(-0.9391f, -0.295707f, -0.612849f, 0.304918f),
new Color(0.632924f, 0.00462222f, 0.268699f, -0.653161f),
new Color(-0.472131f, -0.223032f, -0.190618f, 0.362814f),
new Color(-0.122095f, 0.96746f, 0.863256f, 0.226239f),
new Color(-0.314218f, 0.707067f, -0.196114f, 0.594111f),
new Color(-0.554732f, -0.691444f, -0.183872f, -0.697549f),
new Color(0.201811f, -0.271983f, -0.0972042f, 0.0268987f),
new Color(0.424313f, 0.831922f, 0.176789f, 0.705841f),
new Color(0.415163f, -0.765154f, -0.463756f, -0.850925f),
new Color(0.418529f, -0.584622f, 0.0585436f, -0.634912f),
new Color(-0.215698f, -0.290641f, -0.00109148f, 0.443444f),
new Color(-0.311884f, 0.910497f, 0.813603f, -0.413595f),
new Color(-0.464519f, 0.658968f, -0.487862f, 0.378909f),
new Color(-0.625874f, -0.0805843f, -0.463532f, -0.497419f),
new Color(0.364927f, -0.174177f, 0.0578728f, 0.0917451f),
new Color(0.286893f, 0.867409f, -0.693403f, 0.658919f),
new Color(0.651149f, -0.28131f, -0.0742139f, 0.629413f),
new Color(0.0797535f, -0.908808f, 0.05358f, -0.531914f),
new Color(-0.235643f, -0.20232f, 0.0556322f, 0.255834f),
new Color(0.123746f, 0.798342f, 0.697576f, -0.649831f),
new Color(-0.724099f, 0.486805f, -0.685795f, 0.0627879f),
new Color(-0.463208f, -0.701531f, -0.156876f, -0.620537f),
new Color(0.315751f, -0.0617081f, 0.276996f, 0.261619f),
new Color(0.505891f, 0.649722f, 0.796297f, 0.0411308f),
new Color(0.316298f, -0.815927f, -0.289201f, 0.532078f),
new Color(0.475114f, -0.471477f, 0.56104f, 0.226622f),
new Color(-0.303232f, 0.189555f, 0.202558f, 0.452595f),
new Color(0.713399f, 0.445579f, 0.712816f, 0.166304f),
new Color(-0.970317f, -0.130735f, -0.760568f, -0.251446f),
new Color(-0.426013f, 0.0991545f, 0.138662f, -0.410823f),
new Color(-0.152872f, -0.0611568f, 0.485913f, 0.375226f),
new Color(-0.0638682f, -0.946722f, -0.84746f, 0.176791f),
new Color(0.948536f, 0.0987462f, 0.823804f, -0.134835f),
new Color(0.158957f, 0.876542f, 0.463665f, 0.473959f),
new Color(0.512668f, -0.220553f, 0.0471247f, -0.400378f),
new Color(-0.834045f, -0.512838f, -0.803006f, 0.335161f),
new Color(0.886046f, 0.322539f, 0.762233f, -0.281205f),
new Color(-0.706212f, 0.295994f, -0.00943625f, 0.568246f),
new Color(0.368445f, 0.181919f, 0.216772f, -0.171569f),
new Color(0.386233f, -0.916521f, -0.830085f, 0.0299331f),
new Color(0.343517f, -0.703712f, 0.511238f, -0.57413f),
new Color(-0.0494295f, 0.891983f, 0.369259f, 0.391923f),
new Color(-0.439337f, -0.141093f, -0.309884f, -0.25661f),
new Color(-0.677812f, -0.698545f, -0.113314f, -0.762226f),
new Color(-0.258259f, 0.8383f, 0.297787f, 0.94899f),
new Color(-0.608749f, 0.467308f, -0.181375f, 0.459943f),
new Color(0.372844f, -0.375021f, -0.0504034f, -0.25764f),
new Color(0.556095f, -0.764593f, -0.649529f, 0.55634f),
new Color(0.696083f, -0.419751f, 0.714016f, 0.623071f),
new Color(0.597731f, 0.321198f, 0.493817f, 0.0945984f),
new Color(-0.388182f, -0.0320591f, 0.0659186f, -0.215083f),
new Color(-0.351083f, -0.803275f, 0.731655f, -0.53315f),
new Color(-0.702351f, 0.39579f, -0.553945f, -0.520093f),
new Color(-0.0985173f, 0.715644f, -0.418399f, 0.312792f),
new Color(0.269045f, 0.046643f, -0.208534f, 0.136142f),
new Color(0.0355825f, -0.832147f, -0.343534f, 0.807035f),
new Color(0.877079f, -0.046791f, 0.557949f, 0.564786f),
new Color(0.26901f, 0.698421f, 0.130127f, 0.320372f),
new Color(-0.279592f, -0.520049f, -0.0846106f, 0.222556f),
new Color(-0.849124f, -0.295563f, -0.689654f, -0.398834f),
new Color(0.0446618f, 0.930492f, 0.275251f, -0.51868f),
new Color(-0.442449f, 0.567173f, -0.549746f, -0.269364f),
new Color(-0.0666499f, -0.606093f, -0.0400313f, 0.111795f),
new Color(-0.868659f, -0.0749633f, -0.731376f, -0.0208135f),
new Color(0.653766f, 0.751007f, 0.574547f, 0.459129f),
new Color(0.574912f, 0.136611f, -0.576111f, 0.0539645f),
new Color(-0.152392f, -0.517691f, 0.125685f, 0.128952f),
new Color(-0.0353561f, 0.982877f, 0.870464f, 0.0944777f),
new Color(-0.982948f, 0.14697f, -0.650506f, -0.261056f),
new Color(0.120419f, -0.73798f, -0.555518f, 0.152165f),
new Color(0.0596424f, 0.695616f, -0.26291f, 0.267887f),
new Color(0.937989f, 0.201347f, 0.543396f, -0.683312f),
new Color(-0.802964f, -0.434031f, -0.456562f, 0.477057f),
new Color(0.620887f, -0.476886f, 0.235388f, -0.379955f),
new Color(-0.372613f, -0.128892f, -0.307713f, 0.0258262f),
new Color(-0.221579f, 0.925743f, 0.761374f, 0.36313f),
new Color(-0.621634f, 0.688246f, -0.13617f, 0.797617f),
new Color(0.152416f, -0.599041f, -0.141173f, -0.394924f),
new Color(0.34872f, 0.623946f, 0.411323f, -0.0424169f),
new Color(0.383874f, 0.91353f, -0.568206f, 0.758399f),
new Color(-0.246324f, -0.877143f, -0.0754137f, -0.849003f),
new Color(0.735508f, -0.13527f, 0.682898f, 0.077163f),
new Color(0.0876591f, 0.57344f, 0.150702f, 0.228511f),
new Color(-0.819699f, 0.557029f, 0.150089f, -0.963885f),
new Color(-0.770091f, 0.235316f, -0.398523f, -0.626043f),
new Color(-0.416396f, -0.413236f, -0.496863f, 0.218345f),
new Color(0.345394f, 0.480788f, 0.37176f, 0.0547349f),
new Color(0.632562f, 0.674623f, -0.888604f, 0.441915f),
new Color(0.60272f, -0.584004f, 0.42518f, 0.716363f),
new Color(0.487053f, -0.371063f, 0.475597f, -0.123594f),
new Color(-0.308388f, 0.118496f, 0.296476f, -0.282476f),
new Color(-0.478168f, 0.796168f, 0.173193f, -0.852771f),
new Color(-0.789007f, -0.156204f, -0.323493f, -0.683886f),
new Color(-0.22578f, -0.439343f, -0.0588817f, -0.35101f),
new Color(0.252684f, 0.536626f, 0.119631f, -0.102366f),
new Color(0.808661f, 0.462293f, 0.655407f, 0.545508f),
new Color(-0.520965f, -0.776844f, -0.351505f, 0.60772f),
new Color(0.62719f, -0.1498f, 0.3453f, 0.753701f),
new Color(0.0996658f, 0.451025f, -0.139994f, -0.133504f),
new Color(0.883908f, -0.34765f, 0.747662f, 0.265877f),
new Color(-0.62088f, -0.605165f, -0.607496f, -0.182608f),
new Color(-0.534761f, -0.103109f, 0.508763f, 0.305848f),
new Color(-0.0893018f, 0.396821f, 0.125488f, 0.0464903f),
new Color(-0.520634f, -0.717222f, -0.196162f, -0.777845f),
new Color(-0.756526f, -0.420167f, -0.478558f, -0.369132f),
new Color(-0.164438f, -0.292072f, 0.354785f, -0.752619f),
new Color(0.231389f, -0.426549f, 0.394185f, -0.170191f),
new Color(0.717765f, -0.113043f, -0.623928f, 0.37899f),
new Color(-0.274305f, 0.20759f, -0.434663f, 0.506112f),
new Color(-0.0901381f, 0.607346f, 0.278955f, 0.378649f),
new Color(0.758524f, 0.265049f, 0.304836f, 0.647886f),
new Color(-0.668142f, -0.742429f, -0.00865537f, -0.700623f),
new Color(-0.961146f, -0.0642822f, -0.57004f, -0.0560714f),
new Color(-0.0635112f, -0.0131527f, 0.253128f, -0.595754f),
new Color(0.750906f, -0.615181f, 0.640768f, -0.308249f),
new Color(-0.784085f, 0.121654f, -0.96858f, 0.215616f),
new Color(-0.233537f, 0.399263f, -0.286013f, 0.568769f),
new Color(0.00616717f, 0.983265f, 0.479375f, 0.0833476f),
new Color(0.845983f, 0.525948f, 0.267473f, 0.822547f),
new Color(-0.337562f, -0.813363f, -0.390798f, -0.719016f),
new Color(-0.589737f, -0.360535f, -0.220019f, -0.463185f),
new Color(-0.333257f, -0.135577f, 0.373146f, -0.917981f),
new Color(0.12497f, -0.383418f, 0.0285416f, -0.213535f),
new Color(0.940276f, -0.332097f, -0.540248f, 0.28265f),
new Color(-0.336462f, 0.0937837f, -0.736028f, 0.665236f),
new Color(0.124356f, 0.0530181f, 0.177782f, 0.43353f),
new Color(0.661098f, 0.105986f, 0.596201f, 0.567295f),
new Color(-0.739483f, -0.59441f, -0.0407092f, -0.827494f),
new Color(-0.848214f, -0.233029f, -0.684347f, 0.00963473f),
new Color(-0.0212628f, -0.0908234f, 0.127516f, -0.501042f),
new Color(0.61092f, -0.550682f, 0.78399f, -0.319341f),
new Color(0.575021f, -0.000227153f, -0.817509f, 0.265927f),
new Color(-0.367033f, 0.36814f, 0.0234512f, 0.430656f),
new Color(-0.152405f, 0.897535f, 0.343402f, 0.174594f),
new Color(0.775461f, 0.432277f, 0.129328f, 0.880268f),
new Color(-0.263361f, -0.964393f, -0.447316f, -0.524119f),
new Color(0.0115211f, -0.568337f, -0.295014f, -0.369592f),
new Color(-0.424099f, 0.0183569f, 0.48589f, -0.843544f),
new Color(0.152429f, -0.263616f, 0.229407f, -0.105156f),
new Color(0.848169f, -0.232981f, -0.514514f, 0.148131f),
new Color(-0.149723f, 0.104132f, -0.617075f, 0.651238f),
new Color(0.0160875f, 0.560781f, 0.362453f, 0.533424f),
new Color(0.805342f, 0.142226f, 0.533887f, 0.666852f),
new Color(-0.864273f, -0.498704f, -0.0372436f, -0.965673f),
new Color(-0.962316f, -0.258107f, -0.793588f, -0.0603248f),
new Color(0.246608f, -0.958568f, 0.103704f, -0.743494f),
new Color(0.49552f, -0.495391f, 0.782735f, -0.445391f),
new Color(-0.983863f, 0.0856506f, -0.847384f, 0.407954f),
new Color(-0.554737f, 0.526823f, -0.147676f, 0.474833f),
new Color(-0.378856f, 0.90354f, 0.224593f, 0.198342f),
new Color(0.655946f, 0.394111f, 0.173836f, 0.708141f),
new Color(-0.412297f, -0.879195f, -0.318923f, -0.603045f),
new Color(-0.553429f, -0.255481f, -0.0177949f, -0.373541f),
new Color(-0.247338f, 0.00223398f, 0.505072f, -0.641561f),
new Color(0.320994f, -0.253227f, 0.345613f, 0.0136698f),
new Color(0.897231f, -0.0488473f, -0.621486f, 0.135411f),
new Color(0.0137017f, 0.155106f, -0.519717f, 0.844918f),
new Color(0.0158036f, 0.722406f, 0.507172f, 0.470643f),
new Color(0.931338f, 0.0553329f, 0.500412f, 0.774287f),
new Color(-0.57696f, -0.570787f, -0.154461f, -0.874626f),
new Color(-0.885986f, -0.384347f, -0.658118f, -0.195729f),
new Color(0.084196f, -0.901699f, 0.227605f, -0.818709f),
new Color(0.644695f, -0.763904f, 0.491744f, -0.350533f),
new Color(0.553066f, -0.198112f, -0.839553f, 0.512821f),
new Color(-0.694435f, 0.47309f, -0.102818f, 0.370389f),
new Color(-0.339432f, 0.789382f, 0.0864424f, 0.294176f),
new Color(0.596073f, 0.245477f, 0.135493f, 0.573429f),
new Color(-0.561841f, -0.812726f, -0.156461f, -0.596959f),
new Color(-0.755691f, -0.298912f, -0.36788f, -0.245688f),
new Color(-0.149114f, -0.128247f, 0.393056f, -0.603903f),
new Color(0.371749f, -0.396391f, 0.459332f, -0.127841f),
new Color(0.795536f, -0.0168012f, -0.670696f, 0.239092f),
new Color(-0.125551f, 0.238316f, -0.408635f, 0.628286f),
new Color(-0.0683365f, 0.787144f, 0.379351f, 0.390179f),
new Color(0.86273f, 0.306852f, 0.40664f, 0.713552f),
new Color(0.568713f, 0.80626f, -0.289652f, 0.942775f),
new Color(0.961122f, -0.17061f, 0.366931f, 0.644062f),
new Color(-0.0464919f, 0.738377f, -0.931046f, 0.135011f),
new Color(-0.929104f, -0.334548f, -0.67133f, -0.677892f),
new Color(-0.60317f, -0.217088f, 0.142045f, -0.914209f),
new Color(-0.105123f, -0.831568f, 0.176932f, -0.463811f),
new Color(-0.18384f, -0.405983f, -0.392898f, -0.0834985f),
new Color(-0.229289f, 0.340118f, 0.395501f, 0.0579432f),
new Color(0.780495f, 0.552574f, 0.0582505f, 0.903205f),
new Color(0.909567f, -0.192483f, 0.673625f, -0.70895f),
new Color(0.440332f, -0.896098f, -0.081775f, 0.686653f),
new Color(-0.899593f, -0.26787f, -0.736463f, -0.348835f),
new Color(0.65513f, -0.253839f, 0.629869f, 0.175138f),
new Color(0.0290312f, -0.766451f, -0.105919f, -0.544774f),
new Color(-0.324801f, -0.314313f, -0.394389f, 0.0917082f),
new Color(-0.092526f, 0.0434062f, 0.201457f, 0.0799297f),
new Color(-0.116291f, 0.988786f, 0.467027f, 0.692064f),
new Color(0.867577f, 0.186147f, -0.283804f, 0.76381f),
new Color(0.673169f, -0.561854f, -0.508319f, 0.482571f),
new Color(-0.809782f, 0.0133443f, -0.611619f, -0.782665f),
new Color(-0.500594f, 0.209262f, 0.0331641f, -0.944159f),
new Color(0.16901f, -0.701576f, 0.387451f, -0.275478f),
new Color(0.194093f, 0.616721f, -0.368095f, 0.152621f),
new Color(-0.165053f, 0.285978f, 0.0896829f, 0.389919f),
new Color(0.930436f, 0.34306f, -0.459411f, 0.828797f),
new Color(0.859777f, 0.0203625f, 0.775832f, -0.383805f),
new Color(-0.95863f, 0.273802f, -0.705022f, 0.293919f),
new Color(-0.88104f, -0.435944f, -0.640786f, 0.0528709f),
new Color(-0.518393f, -0.582687f, 0.324873f, -0.663965f),
new Color(-0.13644f, -0.757422f, -0.419367f, -0.461031f),
new Color(0.046827f, -0.350986f, -0.309691f, 0.339847f),
new Color(-0.0977336f, 0.136224f, 0.457786f, 0.148556f),
new Color(0.197517f, 0.896928f, 0.661294f, 0.53858f),
new Color(-0.810306f, 0.569731f, 0.577632f, 0.411835f),
new Color(0.550722f, -0.686525f, -0.761218f, 0.23011f),
new Color(-0.811121f, -0.143302f, -0.339725f, -0.876529f),
new Color(-0.535992f, -0.312569f, 0.219081f, -0.74465f),
new Color(0.00659072f, -0.64082f, 0.414283f, -0.220141f),
new Color(-0.292731f, -0.250521f, -0.203418f, -0.0422367f),
new Color(-0.0993569f, 0.434193f, 0.286183f, 0.20403f),
new Color(0.992153f, 0.0743468f, -0.566431f, 0.78126f),
new Color(0.797209f, 0.21744f, 0.787107f, -0.253761f),
new Color(-0.643356f, 0.512744f, -0.190808f, 0.598361f),
new Color(-0.561016f, 0.34859f, -0.669239f, -0.125044f),
new Color(0.555794f, -0.470537f, 0.40074f, -0.525747f),
new Color(0.491512f, -0.273126f, 0.177071f, -0.398844f),
new Color(0.159947f, -0.189643f, -0.145087f, 0.524439f),
new Color(-0.00707448f, -0.0392305f, 0.400487f, 0.184308f),
new Color(0.324667f, 0.859571f, 0.223478f, 0.779614f),
new Color(0.807014f, -0.527784f, -0.891326f, 0.452907f),
new Color(0.301164f, -0.909036f, -0.889176f, -0.0674204f),
new Color(-0.816081f, -0.287706f, -0.199937f, -0.950307f),
new Color(-0.267673f, -0.832979f, 0.577821f, -0.111129f),
new Color(-0.304903f, -0.534749f, 0.235665f, -0.361318f),
new Color(-0.203171f, -0.193967f, -0.0766228f, -0.121202f),
new Color(0.0101293f, 0.482834f, 0.324833f, 0.321677f),
new Color(0.561871f, 0.733839f, -0.0127481f, 0.819181f),
new Color(0.789682f, 0.0458971f, 0.739125f, -0.0410523f),
new Color(-0.348799f, 0.650166f, -0.369316f, 0.516084f),
new Color(-0.615441f, 0.196341f, -0.460467f, 0.340132f),
new Color(-0.500025f, -0.0651895f, 0.311945f, -0.596618f),
new Color(0.51949f, -0.114714f, -0.415022f, -0.386627f),
new Color(0.303658f, -0.0634188f, 0.540371f, 0.278783f),
new Color(0.0815184f, 0.6059f, 0.404543f, 0.453514f),
new Color(0.786762f, 0.479498f, -0.64108f, 0.703417f),
new Color(0.677867f, 0.344559f, 0.548566f, -0.834648f),
new Color(0.30777f, 0.580065f, -0.730994f, 0.161171f),
new Color(-0.734053f, -0.52895f, -0.457595f, -0.702714f),
new Color(-0.371627f, -0.638093f, 0.434431f, -0.396134f),
new Color(0.0765028f, -0.587917f, 0.524517f, 0.0391147f),
new Color(0.102967f, -0.150392f, -0.241872f, 0.270731f),
new Color(0.0629101f, 0.198161f, 0.16853f, 0.494991f),
new Color(-0.346054f, -0.937627f, 0.355619f, -0.840231f),
new Color(-0.902238f, 0.244498f, -0.275259f, -0.765119f),
new Color(0.133946f, -0.825126f, 0.908349f, -0.130725f),
new Color(0.502877f, 0.852827f, 0.413377f, 0.783327f),
new Color(0.665227f, 0.0114865f, -0.275611f, 0.682995f),
new Color(-0.210493f, 0.536165f, -0.419929f, 0.186613f),
new Color(0.180227f, 0.319124f, 0.305882f, -0.53547f),
new Color(0.038879f, -0.419759f, -0.191655f, 0.0487273f),
new Color(-0.971227f, -0.182077f, -0.0249619f, -0.887791f),
new Color(-0.744568f, 0.524168f, -0.565768f, 0.713333f),
new Color(0.111769f, 0.981221f, 0.159822f, -0.767284f),
new Color(0.730114f, 0.498047f, 0.52855f, 0.570606f),
new Color(-0.571224f, 0.607293f, -0.697899f, 0.0792036f),
new Color(0.289556f, 0.465122f, 0.0385197f, 0.35889f),
new Color(0.483718f, -0.414439f, 0.127843f, -0.623419f),
new Color(-0.127071f, -0.423628f, -0.166887f, 0.164216f),
new Color(0.213398f, -0.891157f, -0.650986f, -0.570274f),
new Color(-0.796815f, -0.346481f, 0.499603f, -0.724163f),
new Color(-0.492628f, 0.744291f, 0.65089f, -0.4826f),
new Color(0.741449f, 0.335859f, 0.336544f, 0.723372f),
new Color(0.591312f, -0.257869f, -0.147812f, 0.671503f),
new Color(-0.303764f, 0.39149f, -0.500501f, 0.00794733f),
new Color(-0.208843f, -0.666328f, 0.359857f, -0.330127f),
new Color(0.163093f, -0.334806f, 0.158195f, -0.115681f),
new Color(-0.931274f, -0.122026f, 0.865573f, -0.492706f),
new Color(-0.869362f, 0.0404603f, -0.773034f, 0.317026f),
new Color(0.98446f, 0.165393f, 0.916007f, 0.21328f),
new Color(0.610107f, 0.701136f, 0.719783f, 0.15621f),
new Color(0.447663f, 0.497268f, -0.358138f, 0.567881f),
new Color(0.228902f, 0.473092f, 0.251611f, 0.29526f),
new Color(-0.0737003f, 0.203841f, 0.309731f, -0.379686f),
new Color(-0.226057f, -0.349135f, -0.122644f, -0.0496535f),
new Color(-0.242177f, -0.896015f, -0.86824f, -0.320321f),
new Color(0.932878f, -0.262619f, -0.580859f, -0.478021f),
new Color(-0.306266f, 0.843115f, 0.876196f, 0.392039f),
new Color(0.653073f, 0.458779f, 0.170602f, 0.801754f),
new Color(0.48483f, 0.384208f, -0.556508f, 0.417975f),
new Color(-0.0404515f, 0.42338f, -0.515617f, -0.180307f),
new Color(0.438545f, -0.469464f, 0.365993f, -0.100554f),
new Color(0.272279f, -0.190952f, 0.0525007f, -0.0612385f),
new Color(-0.922223f, 0.0182384f, 0.615377f, -0.689113f),
new Color(-0.786239f, -0.225259f, -0.812845f, 0.192308f),
new Color(0.963091f, 0.00160575f, -0.0628392f, -0.730174f),
new Color(0.658395f, -0.159962f, 0.656411f, 0.275515f),
new Color(-0.439283f, 0.710256f, -0.618841f, 0.300665f),
new Color(-0.570294f, -0.134997f, -0.351436f, 0.22656f),
new Color(-0.390403f, -0.326302f, 0.058899f, -0.483667f),
new Color(-0.317913f, 0.0419168f, 0.00935316f, 0.0696694f),
new Color(-0.493684f, -0.776127f, -0.211817f, -0.830083f),
new Color(-0.677603f, 0.589031f, 0.869863f, -0.318803f),
new Color(-0.0609034f, 0.934982f, 0.773492f, 0.614108f),
new Color(0.276839f, 0.94738f, -0.223233f, 0.865841f),
new Color(0.0813749f, 0.720691f, -0.489548f, 0.411186f),
new Color(0.37751f, 0.290876f, -0.301917f, 0.151478f),
new Color(0.363202f, 0.111746f, 0.130423f, -0.435471f),
new Color(0.0876862f, -0.222996f, 0.268756f, -0.00645268f),
new Color(-0.58693f, -0.699002f, 0.449231f, -0.777607f),
new Color(-0.79056f, 0.38048f, -0.74867f, 0.00543165f),
new Color(0.564258f, -0.604846f, 0.799556f, -0.0846266f),
new Color(0.719931f, 0.0829735f, 0.563257f, -0.317645f),
new Color(0.547705f, 0.139747f, -0.032935f, 0.626256f),
new Color(-0.53229f, -0.416738f, 0.484674f, -0.0580102f),
new Color(-0.221478f, -0.550583f, -0.402525f, -0.178016f),
new Color(-0.106348f, -0.198913f, 0.116346f, 0.12463f),
new Color(-0.802175f, -0.477828f, 0.865149f, -0.417853f),
new Color(-0.52212f, -0.651163f, -0.397588f, 0.840555f),
new Color(-0.435427f, -0.61566f, 0.687335f, 0.674649f),
new Color(0.187817f, 0.963872f, -0.108212f, 0.842958f),
new Color(0.122493f, 0.674564f, -0.56344f, 0.19556f),
new Color(-0.16777f, 0.365024f, -0.372205f, -0.550269f),
new Color(0.0792916f, -0.663401f, -0.121552f, -0.483319f),
new Color(-0.318376f, -0.0516109f, 0.189588f, 0.143858f),
new Color(0.357372f, 0.924302f, -0.746269f, 0.59287f),
new Color(0.741276f, -0.562351f, 0.0454073f, 0.829894f),
new Color(-0.160437f, 0.798514f, -0.853923f, 0.0881177f),
new Color(-0.6794f, -0.483612f, -0.287051f, -0.653743f),
new Color(-0.468495f, 0.614154f, 0.692125f, -0.381153f),
new Color(0.426783f, -0.324609f, 0.509245f, 0.194788f),
new Color(-0.240635f, -0.301139f, -0.26086f, 0.487637f),
new Color(-0.064405f, 0.312069f, 0.0379354f, -0.302632f),
new Color(0.938867f, 0.264806f, 0.263647f, 0.887493f),
new Color(0.565901f, -0.751822f, 0.167932f, -0.973627f),
new Color(-0.141401f, -0.985358f, -0.221415f, 0.769846f),
new Color(-0.65054f, -0.394734f, -0.640088f, -0.288495f),
new Color(0.564627f, -0.543441f, 0.726539f, -0.241642f),
new Color(-0.283213f, -0.456503f, -0.111079f, -0.346132f),
new Color(-0.417939f, 0.422006f, 0.0974705f, 0.506348f),
new Color(0.163552f, 0.220348f, -0.0487449f, -0.27158f),
new Color(-0.210009f, 0.950011f, 0.630505f, 0.623701f),
new Color(0.851971f, 0.248021f, -0.521715f, 0.677557f),
new Color(0.418249f, -0.838518f, -0.696728f, 0.371749f),
new Color(-0.719485f, -0.226617f, 0.284496f, -0.747904f),
new Color(-0.47948f, 0.564692f, 0.593173f, -0.404854f),
new Color(0.549694f, -0.0557708f, 0.466253f, 0.299493f),
new Color(0.0602907f, 0.658929f, -0.378649f, 0.292336f),
new Color(-0.240298f, 0.14944f, -0.287167f, -0.183182f),
new Color(0.977384f, -0.0998337f, -0.926574f, 0.3298f),
new Color(0.863014f, 0.108217f, 0.822413f, -0.145938f),
new Color(-0.79732f, -0.525537f, -0.734406f, -0.658822f),
new Color(-0.322136f, -0.716063f, -0.644276f, -0.0442569f),
new Color(-0.35586f, -0.466993f, 0.331166f, -0.474019f),
new Color(-0.0587682f, -0.470464f, -0.452138f, -0.113825f),
new Color(0.244493f, -0.26976f, -0.0733278f, 0.479328f),
new Color(0.286875f, 0.0970031f, 0.133361f, -0.0112861f),
new Color(0.644544f, 0.757735f, 0.81532f, 0.352079f),
new Color(-0.835762f, 0.33245f, 0.608089f, 0.520694f),
new Color(-0.0875593f, -0.951936f, -0.817576f, -0.399586f),
new Color(-0.504754f, -0.503745f, -0.104316f, -0.676971f),
new Color(-0.447534f, -0.29046f, 0.648571f, -0.0587797f),
new Color(0.291512f, -0.317352f, 0.23017f, 0.671882f),
new Color(-0.332918f, 0.456572f, -0.451173f, 0.124621f),
new Color(-0.24838f, 0.0887431f, -0.253071f, -0.0882109f),
new Color(0.947107f, 0.117005f, -0.765484f, 0.47052f),
new Color(0.710329f, 0.433647f, 0.733045f, -0.331414f),
new Color(-0.88927f, -0.180691f, 0.101928f, 0.800721f),
new Color(-0.58001f, 0.47635f, -0.581327f, 0.0124136f),
new Color(0.501913f, -0.558528f, 0.681345f, 0.210709f),
new Color(0.452759f, 0.581302f, 0.414533f, -0.035832f),
new Color(0.241494f, 0.54605f, 0.100621f, 0.435381f),
new Color(0.340971f, -0.156472f, -0.0389503f, -0.198393f),
new Color(0.424248f, 0.845332f, -0.0107531f, 0.893405f),
new Color(0.660576f, -0.630106f, -0.872555f, 0.160027f),
new Color(-0.485497f, -0.843281f, -0.601022f, -0.628191f),
new Color(0.0742984f, -0.837546f, 0.4029f, -0.666821f),
new Color(-0.0600544f, -0.593181f, 0.24386f, -0.50923f),
new Color(-0.441952f, -0.229057f, 0.456258f, 0.226044f),
new Color(-0.457487f, 0.268823f, -0.315552f, 0.275939f),
new Color(-0.0158007f, 0.259956f, -0.0343244f, 0.116674f),
new Color(0.859307f, 0.473485f, -0.401863f, 0.772943f),
new Color(0.718976f, -0.448528f, 0.763217f, -0.180825f),
new Color(-0.619806f, 0.567476f, -0.742841f, -0.133182f),
new Color(-0.708738f, -0.0498849f, -0.369077f, 0.718837f),
new Color(-0.542687f, 0.0723764f, -0.0422561f, -0.533841f),
new Color(0.582977f, 0.330099f, -0.217042f, 0.692414f),
new Color(0.34761f, 0.444793f, 0.312791f, 0.261854f),
new Color(0.212905f, 0.00688744f, 0.0403178f, -0.141836f),
new Color(0.686058f, 0.596588f, -0.999228f, 0.0212976f),
new Color(0.339065f, 0.784053f, 0.0952522f, -0.974563f),
new Color(0.28287f, 0.759638f, -0.406697f, -0.781845f),
new Color(0.291245f, -0.848448f, 0.23217f, -0.664448f),
new Color(0.183448f, -0.578055f, 0.598754f, 0.0978366f),
new Color(0.470151f, -0.179208f, -0.130045f, 0.734763f),
new Color(-0.0781646f, 0.540293f, 0.153689f, 0.377263f),
new Color(0.103046f, -0.30694f, 0.0666584f, 0.0143229f)
        };

        #endregion
    }


#if UNITY_EDITOR

    [CustomEditor(typeof(VAOEffectCommandBuffer))]
    public class VAOEffectEditorCmdBuffer : VAOEffectEditor { }

    public class VAOEffectEditor : Editor
    {

        #region Labels

        private readonly GUIContent[] qualityTexts = new GUIContent[6] {
                    new GUIContent("Very Low (2 samples)"),
                    new GUIContent("Low (4 samples)"),
                    new GUIContent("Medium (8 samples)"),
                    new GUIContent("High (16 samples)"),
                    new GUIContent("Very High (32 samples)"),
                    new GUIContent("Ultra (64 samples)")
                };
        private readonly int[] qualityInts = new int[6] { 2, 4, 8, 16, 32, 64 };

        private readonly GUIContent[] qualityTextsAdaptive = new GUIContent[4] {
                    new GUIContent("Low (4 samples) Adaptive"),
                    new GUIContent("Medium (8 samples) Adaptive"),
                    new GUIContent("High (16 samples) Adaptive"),
                    new GUIContent("Very High (32 samples) Adaptive")
                };
        private readonly int[] qualityIntsAdaptive = new int[4] { 4, 8, 16, 32 };

        private readonly GUIContent[] downsamplingTexts = new GUIContent[3] {
                    new GUIContent("Off"),
                    new GUIContent("2x"),
                    new GUIContent("4x")
                };
        private readonly int[] downsamplingInts = new int[3] { 1, 2, 4 };

        private readonly GUIContent[] hzbTexts = new GUIContent[3] {
                    new GUIContent("Off"),
                    new GUIContent("On"),
                    new GUIContent("Auto")
                };
        private readonly int[] hzbInts = new int[3] { 0, 1, 2 };

        private readonly GUIContent[] giTexts = new GUIContent[3] {
                    new GUIContent("Normal"),
                    new GUIContent("Half"),
                    new GUIContent("Quarter")
                };
        private readonly int[] giInts = new int[3] { 1, 2, 4 };

        private readonly GUIContent radiusLabelContent = new GUIContent("Radius:", "Distance of the objects that are still considered for occlusions");
        private readonly GUIContent powerLabelContent = new GUIContent("Power:", "Strength of the occlusion");
        private readonly GUIContent presenceLabelContent = new GUIContent("Presence:", "Increase to make effect more pronounced in corners");
        private readonly GUIContent thicknessLabelContent = new GUIContent("Thickness:", "Thickness of occlusion behind what is seen by camera - controls halo around objects and 'depth' of occlusion behind objects.");
        private readonly GUIContent bordersLabelContent = new GUIContent("Borders AO:", "Amount of occlusion on screen borders - too much can cause flickering or popping when objects enter the screen.");
        private readonly GUIContent ssaoBiasLabelContent = new GUIContent("Bias:", "Bias adjustment raycast algorithm - use to prevent self occlusion.");
        private readonly GUIContent algorithmLabelContent = new GUIContent("Algorithm:", "Algorithm to use - Raycast may work better for some scenes than VAO but may be more noisy. Use what works best.");
        private readonly GUIContent detailLabelContent = new GUIContent("Detail:", "Amount of fine detailed shadows captured by algorithm - more details costs more performance.");

        private readonly GUIContent adaptiveLevelLabelContent = new GUIContent("Adaptive Offset:", "Adjust to fine-tune adaptive sampling quality/performance");
        private readonly GUIContent qualityLabelContent = new GUIContent("Quality:", "Number of samples used");
        private readonly GUIContent downsamplingLabelContent = new GUIContent("Downsampling:", "Reduces the resulting texture size");
        private readonly GUIContent hzbContent = new GUIContent("Hierarchical Buffers:", "Uses downsampled depth and normal buffers for increased performance when using large radius");
        private readonly GUIContent detailQualityLabelContent = new GUIContent("Detail Quality:", "Number of samples used to calculate 'detailed shadows' set by 'Detail' control.");
        private readonly GUIContent detailQualityLabelContentDisabled = new GUIContent("Detail Quality:", "This option is not available when using TEMPORAL FILTERING.");
        
        private readonly GUIContent lumaEnabledLabelContent = new GUIContent("Luma Sensitivity:", "Enables luminance sensitivity");
        private readonly GUIContent lumaThresholdLabelContent = new GUIContent("Threshold:", "Sets which bright surfaces are no longer occluded");
        private readonly GUIContent lumaThresholdHDRLabelContent = new GUIContent("Threshold (HDR):", "Sets which bright surfaces are no longer occluded");
        private readonly GUIContent lumaWidthLabelContent = new GUIContent("Falloff Width:", "Controls the weight of the occlusion as it nears the threshold");
        private readonly GUIContent lumaSoftnessLabelContent = new GUIContent("Falloff Softness:", "Controls the gradient of the falloff");

        private readonly GUIContent effectModeLabelContent = new GUIContent("Effect Mode:", "Switches between different effect modes");
        private readonly GUIContent colorTintLabelContent = new GUIContent("Color Tint:", "Choose the color of the occlusion shadows");

        private readonly GUIContent luminanceModeLabelContent = new GUIContent("Mode:", "Switches sensitivity between luma and HSV value");
        private readonly GUIContent optimizationLabelContent = new GUIContent("Performance Settings:", "");
        private readonly GUIContent cullingPrepassTypeLabelContent = new GUIContent("Downsampled Pre-pass:", "Enable to boost performance, especially on lower radius and higher resolution/quality settings. Greedy option is faster but might produce minute detail loss.");
        private readonly GUIContent cullingPrepassTypeLabelContentDisabled = new GUIContent("Downsampled Pre-pass:", "This option is not available when using TEMPORAL FILTERING.");
        private readonly GUIContent adaptiveSamplingLabelContent = new GUIContent("Adaptive Sampling:", "Automagically sets progressively lower quality for distant geometry");
        private readonly GUIContent temporalFilteringLabelContent = new GUIContent("Temporal Filtering:", "Uses information from previous frames to improve visual quality and performance.");
        
        private readonly GUIContent radiusLimitsFoldoutLabelContent = new GUIContent("Near/Far Occlusion Limits:", "Special occlusion behaviour depending on distance from camera");

        private readonly GUIContent pipelineLabelContent = new GUIContent("Rendering Pipeline:", "Unity Rendering pipeline options.");
        private readonly GUIContent commandBufferLabelContent = new GUIContent("Command Buffer:", "Insert effect via command buffer (BeforeImageEffectsOpaque event)");
        private readonly GUIContent gBufferLabelContent = new GUIContent("G-Buffer Depth&Normals:", "Take depth&normals from GBuffer of deferred rendering path, use this to overcome compatibility issues or for better precision");
        private readonly GUIContent dedicatedDepthBufferLabelContent = new GUIContent("High precision depth buffer:", "Uses higher precision depth buffer (forward path only). This may also fix some materials that work normally in deferred path.");
        private readonly GUIContent cameraEventLabelContent = new GUIContent("Cmd. buffer integration stage:", "Where in rendering pipeline is VAO calculated - only available for DEFERRED rendering path. Before Reflecitons is earliest event and AO is influenced by reflections. After Lighting only influences lighting of the scene. On event Before Images Effect Opaque the VAO is applied to final image.");
        private readonly GUIContent normalsSourceLabelContent = new GUIContent("Normals Source:", "Normals will be taken either from Unity's built-in G-Buffer (preferred way) or recalculated from depth buffer. Recalculation can fix incorrect normals but may be less accurate.");

        private readonly GUIContent blurModeContent = new GUIContent("Blur Mode:", "Switches between different blur effects");
        private readonly GUIContent blurQualityContent = new GUIContent("Blur Quality:", "Switches between faster and less accurate or and slower but more precise blur implementation");
        private readonly GUIContent blurFoldoutContent = new GUIContent("Enhanced Blur Settings", "Lets you control behaviour of the enhanced blur");
        private readonly GUIContent blurSizeContent = new GUIContent("Blur Size:", "Change to adjust the size of area that is averaged");
        private readonly GUIContent blurDeviationContent = new GUIContent("Blur Sharpness:", "Standard deviation for Gaussian blur - smaller deviation means sharper image");

        private readonly GUIContent aoLabelContent = new GUIContent("Output AO only:", "Displays just the occlusion - used for tuning the settings");

        private readonly GUIContent colorBleedLabelContent = new GUIContent("Color Bleed Settings", "Lets you control indirect illumination");
        private readonly GUIContent colorBleedPowerLabelContent = new GUIContent("Power:", "Strength of the color bleed");
        private readonly GUIContent colorBleedPresenceLabelContent = new GUIContent("Presence:", "Smoothly limits maximal saturation of color bleed");
        private readonly GUIContent colorBleedQualityLabelContent = new GUIContent("Quality:", "Samples used for color bleed");
        private readonly GUIContent ColorBleedSelfOcclusionLabelContent = new GUIContent("Dampen Self-Bleeding:", "Limits casting color on itself");
        private readonly GUIContent backfaceLabelContent = new GUIContent("Skip Backfaces:", "Skips surfaces facing other way");
        private readonly GUIContent screenFormatLabelContent = new GUIContent("Intermediate texture format:", "Texture format to use for mixing VAO with scene. Auto is recommended.");
        private readonly GUIContent farPlaneSourceLabelContent = new GUIContent("Far plane source:", "Where to take far plane distance from. Camera is needed for post-processing stack temporal AA compatibility. Use Projection Params option for compatibility with other effects.");
        private readonly GUIContent maxRadiusLabelContent = new GUIContent("Limit Max Radius:", "Maximal radius given as percentage of screen that will be considered for occlusion. Use to avoid performance drop for objects close to camera.");
        private readonly GUIContent maxRadiusSliderContent = new GUIContent("Max Radius:", "Maximal radius given as fraction of the screen that can be considered for occlusion.");
        private readonly GUIContent distanceFalloffModeLabelContent = new GUIContent("Distance Falloff:", "With this enabled occlusion starts to fall off rapidly at certain distance.");
        private readonly GUIContent distanceFalloffAbsoluteLabelContent = new GUIContent("Falloff Start:", "Falloff distance set as an absolute value (same as Far Clipping Plane).");
        private readonly GUIContent distanceFalloffRelativeLabelContent = new GUIContent("Falloff Start:", "Falloff start set relative to occlusion area covering one screen pixel.");
        private readonly GUIContent distanceFalloffSpeedLabelContent = new GUIContent("Falloff Speed:", "How fast the occlusion decreases after the falloff border.");
        private readonly GUIContent colorBleedSameHueAttenuationLabelContent = new GUIContent("Same Color Hue Attenuation", "Attenuates colorbleed thrown on surface of the same color.");
        private readonly GUIContent colorBleedSameHueAttenuationHueFilterLabelContent = new GUIContent("Hue Filter", "Set how much the hue has to differ to be filtered out.");
        private readonly GUIContent colorBleedSameHueAttenuationHueToleranceLabelContent = new GUIContent("Tolerance:", "How much the hue has to differ to be filtered out.");
        private readonly GUIContent colorBleedSameHueAttenuationHueSoftnessLabelContent = new GUIContent("Softness:", "How smooth will the transision be.");
        private readonly GUIContent colorBleedSameHueAttenuationSaturationFilterLabelContent = new GUIContent("Saturation Filter", "Set minimal saturation of color where hue filter will be applied.");
        private readonly GUIContent colorBleedSameHueAttenuationSaturationToleranceLabelContent = new GUIContent("Threshold:", "Saturation threshold when hue filter will be applied.");
        private readonly GUIContent colorBleedSameHueAttenuationSaturationSoftnessLabelContent = new GUIContent("Softness:", "How smooth will the transision be.");
        private readonly GUIContent colorBleedSameHueAttenuationBrightnessLabelContent = new GUIContent("Brightness:", "Limits value component of HSV model of the result.");

        #endregion

        #region Previous Settings Cache

        private float lastLumaThreshold;
        private float lastLumaKneeWidth;
        private float lastLumaKneeLinearity;
        private float lastLumaMaxFx;
        private bool lastLumaSensitive;
        private VAOEffectCommandBuffer.EffectMode lastEffectMode;
        private float lumaMaxFx = 10.0f;

        #endregion

        #region Luma Graph Widget

        private GraphWidget lumaGraphWidget;
        private GraphWidgetDrawingParameters lumaGraphWidgetParams;

        private GraphWidgetDrawingParameters GetLumaGraphWidgetParameters(VAOEffectCommandBuffer vaoScript)
        {
            if (lumaGraphWidgetParams != null &&
                lastLumaThreshold == vaoScript.LumaThreshold &&
                lastLumaKneeWidth == vaoScript.LumaKneeWidth &&
                lastLumaMaxFx == lumaMaxFx &&
                lastLumaKneeLinearity == vaoScript.LumaKneeLinearity) return lumaGraphWidgetParams;

            lastLumaThreshold = vaoScript.LumaThreshold;
            lastLumaKneeWidth = vaoScript.LumaKneeWidth;
            lastLumaKneeLinearity = vaoScript.LumaKneeLinearity;
            lastLumaMaxFx = lumaMaxFx;

            lumaGraphWidgetParams = new GraphWidgetDrawingParameters()
            {
                GraphSegmentsCount = 128,
                GraphColor = Color.white,
                GraphThickness = 2.0f,
                GraphFunction = ((float x) =>
                {
                    float Y = (x - (vaoScript.LumaThreshold - vaoScript.LumaKneeWidth)) * (1.0f / (2.0f * vaoScript.LumaKneeWidth));
                    x = Mathf.Min(1.0f, Mathf.Max(0.0f, Y));
                    return ((-Mathf.Pow(x, vaoScript.LumaKneeLinearity) + 1));
                }),
                YScale = 0.65f,
                MinY = 0.1f,
                MaxFx = lumaMaxFx,
                GridLinesXCount = 4,
                LabelText = "Luminance sensitivity curve",
                Lines = new List<GraphWidgetLine>()
                        {
                            new GraphWidgetLine() {
                                Color = Color.red,
                                Thickness = 2.0f,
                                From = new Vector3(vaoScript.LumaThreshold / lumaMaxFx, 0.0f, 0.0f),
                                To = new Vector3(vaoScript.LumaThreshold / lumaMaxFx, 1.0f, 0.0f)
                            },
                            new GraphWidgetLine() {
                                Color = Color.blue * 0.7f,
                                Thickness = 2.0f,
                                From = new Vector3((vaoScript.LumaThreshold - vaoScript.LumaKneeWidth) / lumaMaxFx, 0.0f, 0.0f),
                                To = new Vector3((vaoScript.LumaThreshold - vaoScript.LumaKneeWidth) / lumaMaxFx, 1.0f, 0.0f)
                            },
                            new GraphWidgetLine() {
                                Color = Color.blue * 0.7f,
                                Thickness = 2.0f,
                                From = new Vector3((vaoScript.LumaThreshold + vaoScript.LumaKneeWidth) / lumaMaxFx, 0.0f, 0.0f),
                                To = new Vector3((vaoScript.LumaThreshold + vaoScript.LumaKneeWidth) / lumaMaxFx, 1.0f, 0.0f)
                            }
                        }
            };

            return lumaGraphWidgetParams;
        }

        #endregion

        private bool isHDR;
        private Camera camera;

        #region VAO Implementation Utilities

        private float GetRadiusForDepthAndScreenRadius(Camera camera, float pixelDepth, float maxRadiusOnScreen)
        {
            return -(pixelDepth * maxRadiusOnScreen) / camera.projectionMatrix.m11;
        }

        private float GetScreenSizeForDepth(Camera camera, float pixelDepth, float radius)
        {
            return -(radius * camera.projectionMatrix.m11) / pixelDepth;
        }

        private float GetScreenSizeForDepth(Camera camera, float pixelDepth, float radius, bool maxRadiusEnabled, float maxRadiusCutoffDepth, float maxRadiusOnScreen)
        {
            if (maxRadiusEnabled && pixelDepth > maxRadiusCutoffDepth)
            {
                radius = GetRadiusForDepthAndScreenRadius(camera, pixelDepth, maxRadiusOnScreen);
            }

            return GetScreenSizeForDepth(camera, pixelDepth, radius);
        }

        private float GetDepthForScreenSize(Camera camera, float sizeOnScreen, float radius)
        {
            return -(radius * camera.projectionMatrix.m11) / sizeOnScreen;
        }

        #endregion

        #region Unity Utilities

        private void SetIcon()
        {
            try
            {
                Texture2D icon = (Texture2D)Resources.Load("wilberforce_script_icon");
                Type editorGUIUtilityType = typeof(UnityEditor.EditorGUIUtility);
                System.Reflection.BindingFlags bindingFlags = System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic;
                object[] args = new object[] { target, icon };
                editorGUIUtilityType.InvokeMember("SetIconForObject", bindingFlags, null, null, args);
            }
            catch (Exception ex)
            {
                if (Debug.isDebugBuild) Debug.Log("VAO Effect Error: There was an exception while setting icon to VAO script: " + ex.Message);
            }
        }

        void OnEnable()
        {
            if (lumaGraphWidget == null) lumaGraphWidget = new GraphWidget();

            VAOEffectCommandBuffer effect = (target as VAOEffectCommandBuffer);
            camera = effect.GetComponent<Camera>();

            isHDR = isCameraHDR(camera);
            SetIcon();

        }

        private bool isCameraHDR(Camera camera)
        {

#if UNITY_5_6_OR_NEWER
            if (camera != null) return camera.allowHDR;
#else
            if (camera != null) return camera.hdr;
#endif
            return false;
        }

        #endregion

        override public void OnInspectorGUI()
        {
            var vaoScript = target as VAOEffectCommandBuffer;
            //vaoScript.vaoShader = EditorGUILayout.ObjectField("Vao Shader", vaoScript.vaoShader, typeof(Shader), false) as UnityEngine.Shader;

            if (vaoScript.ShouldUseHierarchicalBuffer())
            {
                hzbTexts[2].text = "Auto (currently on)";
            }
            else
            {
                hzbTexts[2].text = "Auto (currently off)";
            }

            EditorGUILayout.Space();

            vaoScript.Radius = EditorGUILayout.FloatField(radiusLabelContent, vaoScript.Radius);
            vaoScript.Power = EditorGUILayout.FloatField(powerLabelContent, vaoScript.Power);
            vaoScript.Presence = EditorGUILayout.Slider(presenceLabelContent, vaoScript.Presence, 0.0f, 1.0f);

            if (vaoScript.Algorithm == VAOEffectCommandBuffer.AlgorithmType.StandardVAO)
            {
                vaoScript.DetailAmountVAO = EditorGUILayout.Slider(detailLabelContent, vaoScript.DetailAmountVAO, 0.0f, 1.0f);
            }
            else
            {
                vaoScript.DetailAmountRaycast = EditorGUILayout.Slider(detailLabelContent, vaoScript.DetailAmountRaycast, 0.0f, 1.0f);
            }

            EditorGUILayout.Space();

            if (vaoScript.AdaptiveType == VAOEffectCommandBuffer.AdaptiveSamplingType.Disabled)
            {
                vaoScript.Quality = EditorGUILayout.IntPopup(qualityLabelContent, vaoScript.Quality, qualityTexts, qualityInts);
            }
            else
            {
                vaoScript.Quality = EditorGUILayout.IntPopup(qualityLabelContent, vaoScript.Quality, qualityTextsAdaptive, qualityIntsAdaptive);
            }
            EditorGUILayout.Space();

            vaoScript.Algorithm = (VAOEffectCommandBuffer.AlgorithmType)EditorGUILayout.EnumPopup(algorithmLabelContent, vaoScript.Algorithm);
  
            EditorGUI.indentLevel++;
            if (vaoScript.Algorithm == VAOEffectCommandBuffer.AlgorithmType.StandardVAO)
            {
                vaoScript.Thickness = EditorGUILayout.Slider(thicknessLabelContent, vaoScript.Thickness, 0.01f, 1.0f);
            }
            else
            {
                vaoScript.SSAOBias = EditorGUILayout.Slider(ssaoBiasLabelContent, vaoScript.SSAOBias, 0.0f, 0.1f);
            }

            vaoScript.BordersIntensity = EditorGUILayout.Slider(bordersLabelContent, vaoScript.BordersIntensity, 0.0f, 1.0f);
            EditorGUI.indentLevel--;

            EditorGUILayout.Space();
            EditorGUI.indentLevel++;

            vaoScript.radiusLimitsFoldout = EditorGUI.Foldout(EditorGUILayout.GetControlRect(), vaoScript.radiusLimitsFoldout, radiusLimitsFoldoutLabelContent, true, EditorStyles.foldout);

            if (vaoScript.radiusLimitsFoldout)
            {
                //EditorGUILayout.BeginHorizontal();
                vaoScript.MaxRadiusEnabled = EditorGUILayout.Toggle(maxRadiusLabelContent, vaoScript.MaxRadiusEnabled, customToggleStyle);
                if (vaoScript.MaxRadiusEnabled)
                    vaoScript.MaxRadius = EditorGUILayout.Slider(maxRadiusSliderContent, vaoScript.MaxRadius, 0.05f, 3.0f);

                //EditorGUILayout.EndHorizontal();

                vaoScript.DistanceFalloffMode = (VAOEffectCommandBuffer.DistanceFalloffModeType)EditorGUILayout.EnumPopup(distanceFalloffModeLabelContent, vaoScript.DistanceFalloffMode);
                switch (vaoScript.DistanceFalloffMode)
                {
                    case VAOEffectCommandBuffer.DistanceFalloffModeType.Off:
                        break;
                    case VAOEffectCommandBuffer.DistanceFalloffModeType.Absolute:
                        vaoScript.DistanceFalloffStartAbsolute = EditorGUILayout.FloatField(distanceFalloffAbsoluteLabelContent, vaoScript.DistanceFalloffStartAbsolute);
                        vaoScript.DistanceFalloffSpeedAbsolute = EditorGUILayout.FloatField(distanceFalloffSpeedLabelContent, vaoScript.DistanceFalloffSpeedAbsolute);
                        break;
                    case VAOEffectCommandBuffer.DistanceFalloffModeType.Relative:
                        vaoScript.DistanceFalloffStartRelative = EditorGUILayout.Slider(distanceFalloffRelativeLabelContent, vaoScript.DistanceFalloffStartRelative, 0.0f, 1.0f);
                        vaoScript.DistanceFalloffSpeedRelative = EditorGUILayout.Slider(distanceFalloffSpeedLabelContent, vaoScript.DistanceFalloffSpeedRelative, 0.0f, 1.0f);
                        break;
                    default:
                        break;
                }
                EditorGUILayout.Space();
            }

            vaoScript.optimizationFoldout = EditorGUI.Foldout(EditorGUILayout.GetControlRect(), vaoScript.optimizationFoldout, optimizationLabelContent, true, EditorStyles.foldout);

            if (vaoScript.optimizationFoldout)
            {
                vaoScript.EnableTemporalFiltering = EditorGUILayout.Toggle(temporalFilteringLabelContent, vaoScript.EnableTemporalFiltering);
                vaoScript.AdaptiveType = (VAOEffectCommandBuffer.AdaptiveSamplingType)EditorGUILayout.EnumPopup(adaptiveSamplingLabelContent, vaoScript.AdaptiveType);
                if (vaoScript.AdaptiveType == VAOEffectCommandBuffer.AdaptiveSamplingType.EnabledManual)
                {
                    vaoScript.AdaptiveQualityCoefficient = EditorGUILayout.Slider(adaptiveLevelLabelContent, vaoScript.AdaptiveQualityCoefficient, 0.5f, 2.0f);
                }

                EditorGUI.BeginDisabledGroup(vaoScript.EnableTemporalFiltering);
                vaoScript.CullingPrepassMode = (VAOEffectCommandBuffer.CullingPrepassModeType)EditorGUILayout.EnumPopup(vaoScript.EnableTemporalFiltering ? cullingPrepassTypeLabelContentDisabled : cullingPrepassTypeLabelContent, vaoScript.CullingPrepassMode);
                EditorGUI.EndDisabledGroup();

                vaoScript.Downsampling = EditorGUILayout.IntPopup(downsamplingLabelContent, vaoScript.Downsampling, downsamplingTexts, downsamplingInts);
                vaoScript.HierarchicalBufferState = (VAOEffectCommandBuffer.HierarchicalBufferStateType)EditorGUILayout.IntPopup(hzbContent, (int)vaoScript.HierarchicalBufferState, hzbTexts, hzbInts);

                EditorGUI.BeginDisabledGroup(vaoScript.EnableTemporalFiltering);
                vaoScript.DetailQuality = (VAOEffectCommandBuffer.DetailQualityType)EditorGUILayout.EnumPopup(vaoScript.EnableTemporalFiltering ? detailQualityLabelContentDisabled : detailQualityLabelContent, vaoScript.DetailQuality);
                EditorGUI.EndDisabledGroup();


                EditorGUILayout.Space();
            }

            vaoScript.pipelineFoldout = EditorGUI.Foldout(EditorGUILayout.GetControlRect(), vaoScript.pipelineFoldout, pipelineLabelContent, true, EditorStyles.foldout);
            if (vaoScript.pipelineFoldout)
            {
                vaoScript.CommandBufferEnabled = EditorGUILayout.Toggle(commandBufferLabelContent, vaoScript.CommandBufferEnabled);

                EditorGUILayout.Space();

                vaoScript.NormalsSource = (VAOEffectCommandBuffer.NormalsSourceType)EditorGUILayout.EnumPopup(normalsSourceLabelContent, vaoScript.NormalsSource);

                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Deferred rendering path" + (camera.actualRenderingPath == RenderingPath.DeferredShading ? " (active):" : ":"));
                EditorGUI.indentLevel++;
                EditorGUI.BeginDisabledGroup(!vaoScript.CommandBufferEnabled && vaoScript.VaoCameraEvent != VAOEffectCommandBuffer.VAOCameraEventType.BeforeImageEffectsOpaque);
                vaoScript.VaoCameraEvent = (VAOEffectCommandBuffer.VAOCameraEventType)EditorGUILayout.EnumPopup(cameraEventLabelContent, vaoScript.VaoCameraEvent);
                EditorGUI.EndDisabledGroup();

                DisplayCamerasEventStateMessage(vaoScript);
                vaoScript.UseGBuffer = EditorGUILayout.Toggle(gBufferLabelContent, vaoScript.UseGBuffer);

                DisplayGBufferStateMessage(vaoScript);

                EditorGUI.indentLevel--;
                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Forward rendering path" + (camera.actualRenderingPath == RenderingPath.Forward ? " (active):" : ":"));
                EditorGUI.indentLevel++;
                vaoScript.UsePreciseDepthBuffer = EditorGUILayout.Toggle(dedicatedDepthBufferLabelContent, vaoScript.UsePreciseDepthBuffer);
                EditorGUI.indentLevel--;

                EditorGUILayout.Space();

                vaoScript.FarPlaneSource = (VAOEffectCommandBuffer.FarPlaneSourceType)EditorGUILayout.EnumPopup(farPlaneSourceLabelContent, vaoScript.FarPlaneSource);
            }


            EditorGUI.indentLevel--;
            EditorGUILayout.Space();

            HeaderWithToggle(lumaEnabledLabelContent.text, ref vaoScript.lumaFoldout, ref vaoScript.IsLumaSensitive);

            if (vaoScript.IsLumaSensitive)
            {
                if (lastLumaSensitive == false) vaoScript.lumaFoldout = true;
            }

            EditorGUILayout.Space();
            EditorGUI.indentLevel++;

            if (vaoScript.lumaFoldout)
            {
                EditorGUI.BeginDisabledGroup(!vaoScript.IsLumaSensitive);

                vaoScript.LuminanceMode = (VAOEffectCommandBuffer.LuminanceModeType)EditorGUILayout.EnumPopup(luminanceModeLabelContent, vaoScript.LuminanceMode);

                lumaMaxFx = 1.0f;
                float tresholdMax = lumaMaxFx;
                float kneeWidthMax = lumaMaxFx;
                GUIContent tresholdLabel = lumaThresholdLabelContent;
                if (camera != null)
                {
                    if (isCameraHDR(camera)) //< Check current setting and update isHDR variable if needed
                    {
                        lumaMaxFx = 10.0f;
                        tresholdMax = lumaMaxFx;
                        kneeWidthMax = lumaMaxFx;
                        tresholdLabel = lumaThresholdHDRLabelContent;

                        if (!isHDR)
                        {
                            vaoScript.LumaThreshold *= 10.0f;
                            vaoScript.LumaKneeWidth *= 10.0f;
                            isHDR = true;
                        }
                    }
                    else
                    {
                        if (isHDR)
                        {
                            vaoScript.LumaThreshold *= 0.1f;
                            vaoScript.LumaKneeWidth *= 0.1f;
                            isHDR = false;
                        }
                    }
                }

                vaoScript.LumaThreshold = EditorGUILayout.Slider(tresholdLabel, vaoScript.LumaThreshold, 0.0f, tresholdMax);
                vaoScript.LumaKneeWidth = EditorGUILayout.Slider(lumaWidthLabelContent, vaoScript.LumaKneeWidth, 0.0f, kneeWidthMax);
                vaoScript.LumaKneeLinearity = EditorGUILayout.Slider(lumaSoftnessLabelContent, vaoScript.LumaKneeLinearity, 1.0f, 10.0f);
                EditorGUILayout.Space();
                lumaGraphWidget.Draw(GetLumaGraphWidgetParameters(vaoScript));

                EditorGUI.EndDisabledGroup();
            }

            EditorGUI.indentLevel--;
            EditorGUILayout.Space();

            vaoScript.Mode = (VAOEffectCommandBuffer.EffectMode)EditorGUILayout.EnumPopup(effectModeLabelContent, vaoScript.Mode);

            EditorGUI.indentLevel++;
            switch (vaoScript.Mode)
            {
                case VAOEffectCommandBuffer.EffectMode.Simple:
                    break;
                case VAOEffectCommandBuffer.EffectMode.ColorTint:
                    EditorGUILayout.Space();
                    vaoScript.ColorTint = EditorGUILayout.ColorField(colorTintLabelContent, vaoScript.ColorTint);
                    break;
                case VAOEffectCommandBuffer.EffectMode.ColorBleed:
                    EditorGUILayout.Space();
                    vaoScript.colorBleedFoldout = EditorGUI.Foldout(EditorGUILayout.GetControlRect(), vaoScript.colorBleedFoldout, colorBleedLabelContent, true, EditorStyles.foldout);

                    if (lastEffectMode != VAOEffectCommandBuffer.EffectMode.ColorBleed) vaoScript.colorBleedFoldout = true;

                    if (vaoScript.colorBleedFoldout)
                    {
                        vaoScript.ColorBleedPower = EditorGUILayout.FloatField(colorBleedPowerLabelContent, vaoScript.ColorBleedPower);
                        vaoScript.ColorBleedPresence = EditorGUILayout.Slider(colorBleedPresenceLabelContent, vaoScript.ColorBleedPresence, 0.0f, 1.0f);

                        if (vaoScript.CommandBufferEnabled)
                        {
                            vaoScript.IntermediateScreenTextureFormat = (VAOEffect.ScreenTextureFormat)EditorGUILayout.EnumPopup(screenFormatLabelContent, vaoScript.IntermediateScreenTextureFormat);
                        }

                        vaoScript.ColorbleedHueSuppresionEnabled = EditorGUILayout.ToggleLeft(colorBleedSameHueAttenuationLabelContent, vaoScript.ColorbleedHueSuppresionEnabled);

                        if (EditorGUILayout.BeginFadeGroup(vaoScript.ColorbleedHueSuppresionEnabled ? 1.0f : 0.0f))
                        {
                            EditorGUI.indentLevel++;
                            EditorGUILayout.LabelField(colorBleedSameHueAttenuationHueFilterLabelContent);
                            EditorGUI.indentLevel++;
                            vaoScript.ColorBleedHueSuppresionThreshold = EditorGUILayout.Slider(colorBleedSameHueAttenuationHueToleranceLabelContent, vaoScript.ColorBleedHueSuppresionThreshold, 0.0f, 50.0f);
                            vaoScript.ColorBleedHueSuppresionWidth = EditorGUILayout.Slider(colorBleedSameHueAttenuationHueSoftnessLabelContent, vaoScript.ColorBleedHueSuppresionWidth, 0.0f, 10.0f);
                            EditorGUI.indentLevel--;
                            EditorGUILayout.LabelField(colorBleedSameHueAttenuationSaturationFilterLabelContent);
                            EditorGUI.indentLevel++;
                            vaoScript.ColorBleedHueSuppresionSaturationThreshold = EditorGUILayout.Slider(colorBleedSameHueAttenuationSaturationToleranceLabelContent, vaoScript.ColorBleedHueSuppresionSaturationThreshold, 0.0f, 1.0f);
                            vaoScript.ColorBleedHueSuppresionSaturationWidth = EditorGUILayout.Slider(colorBleedSameHueAttenuationSaturationSoftnessLabelContent, vaoScript.ColorBleedHueSuppresionSaturationWidth, 0.0f, 1.0f);
                            vaoScript.ColorBleedHueSuppresionBrightness = EditorGUILayout.Slider(colorBleedSameHueAttenuationBrightnessLabelContent, vaoScript.ColorBleedHueSuppresionBrightness, 0.0f, 1.0f);
                            EditorGUI.indentLevel--;
                            EditorGUI.indentLevel--;
                            EditorGUILayout.Space();
                        }
                        EditorGUILayout.EndFadeGroup();


                        vaoScript.ColorBleedQuality = EditorGUILayout.IntPopup(colorBleedQualityLabelContent, vaoScript.ColorBleedQuality, giTexts, giInts);
                        vaoScript.ColorBleedSelfOcclusionFixLevel = (VAOEffectCommandBuffer.ColorBleedSelfOcclusionFixLevelType)EditorGUILayout.EnumPopup(ColorBleedSelfOcclusionLabelContent, vaoScript.ColorBleedSelfOcclusionFixLevel);
                        vaoScript.GiBackfaces = EditorGUILayout.Toggle(backfaceLabelContent, vaoScript.GiBackfaces);
                    }
                    break;
                default:
                    break;
            }
            EditorGUI.indentLevel--;

            EditorGUILayout.Space();

            vaoScript.BlurQuality = (VAOEffectCommandBuffer.BlurQualityType)EditorGUILayout.EnumPopup(blurQualityContent, vaoScript.BlurQuality);
            vaoScript.BlurMode = (VAOEffectCommandBuffer.BlurModeType)EditorGUILayout.EnumPopup(blurModeContent, vaoScript.BlurMode);

            if (vaoScript.BlurMode == VAOEffectCommandBuffer.BlurModeType.Enhanced)
            {
                EditorGUILayout.Space();
                EditorGUI.indentLevel++;
                vaoScript.blurFoldout = EditorGUI.Foldout(EditorGUILayout.GetControlRect(), vaoScript.blurFoldout, blurFoldoutContent, true, EditorStyles.foldout);
                if (vaoScript.blurFoldout)
                {
                    vaoScript.EnhancedBlurSize = EditorGUILayout.IntSlider(blurSizeContent, vaoScript.EnhancedBlurSize, 3, 17);
                    vaoScript.EnhancedBlurDeviation = EditorGUILayout.Slider(blurDeviationContent, vaoScript.EnhancedBlurDeviation, 0.01f, 3.0f);
                }
                EditorGUI.indentLevel--;
            }
            EditorGUILayout.Space();

            vaoScript.OutputAOOnly = EditorGUILayout.Toggle(aoLabelContent, vaoScript.OutputAOOnly);
            vaoScript.aboutFoldout = EditorGUI.Foldout(EditorGUILayout.GetControlRect(), vaoScript.aboutFoldout, "About", true, EditorStyles.foldout);
            if (vaoScript.aboutFoldout)
            {
                EditorGUILayout.HelpBox("Volumetric Ambient Occlusion v2.0 by Project Wilberforce.\n\nThank you for your purchase and if you have any questions, issues or suggestions, feel free to contact us at <projectwilberforce@gmail.com>.", MessageType.Info);
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Contact Support"))
                {
                    Application.OpenURL("mailto:projectwilberforce@gmail.com");
                }
                //if (GUILayout.Button("Rate on Asset Store"))
                //{
                //    Application.OpenURL("https://www.assetstore.unity3d.com/en/#!/account/downloads/search=Volumetric%20Ambient%20Occlusion");
                //}
                if (GUILayout.Button("Asset Store Page"))
                {
                    Application.OpenURL("http://u3d.as/xzs"); //< Official Unity shortened link to asset store page of VAO
                }
                EditorGUILayout.EndHorizontal();
            }
            lastLumaSensitive = vaoScript.IsLumaSensitive;
            lastEffectMode = vaoScript.Mode;

            if (GUI.changed)
            {
                // Force parameters to be positive
                vaoScript.Radius = Mathf.Clamp(vaoScript.Radius, 0.001f, float.MaxValue);
                vaoScript.Power = Mathf.Clamp(vaoScript.Power, 0, float.MaxValue);
                vaoScript.ColorBleedPower = Mathf.Clamp(vaoScript.ColorBleedPower, 0, float.MaxValue);

                if (vaoScript.Quality == 64 && vaoScript.AdaptiveType != VAOEffectCommandBuffer.AdaptiveSamplingType.Disabled)
                {
                    vaoScript.Quality = 32;
                }

                if (vaoScript.EnhancedBlurSize % 2 == 0)
                {
                    vaoScript.EnhancedBlurSize += 1;
                }

                // Mark as dirty
                EditorUtility.SetDirty(target);
            }
            Undo.RecordObject(target, "VAO change");
        }

        void DisplayCamerasEventStateMessage(VAOEffectCommandBuffer vaoScript)
        {
            if (vaoScript.Mode == VAOEffectCommandBuffer.EffectMode.ColorBleed &&
                vaoScript.CommandBufferEnabled &&
                vaoScript.VaoCameraEvent != VAOEffectCommandBuffer.VAOCameraEventType.BeforeImageEffectsOpaque)
            {
                EditorGUILayout.HelpBox("Cannot use selected cmd. buffer integration stage. Only BeforeImageEffectsOpaque is supported in color bleeding mode.", MessageType.Warning);
                return;
            }

            if (!vaoScript.CommandBufferEnabled &&
               vaoScript.VaoCameraEvent != VAOEffectCommandBuffer.VAOCameraEventType.BeforeImageEffectsOpaque)
            {
                EditorGUILayout.HelpBox("Cannot use selected cmd. buffer integration stage. Other events than BeforeImageEffectsOpaque are only supported in 'command buffer' mode.", MessageType.Warning);
                return;
            }

            if (vaoScript.CommandBufferEnabled &&
                camera.actualRenderingPath == RenderingPath.Forward &&
               vaoScript.VaoCameraEvent != VAOEffectCommandBuffer.VAOCameraEventType.BeforeImageEffectsOpaque)
            {
                EditorGUILayout.HelpBox("Cannot use selected cmd. buffer integration stage. Other events than BeforeImageEffectsOpaque are only supported 'deferred' rendering path.", MessageType.Warning);
            }
        }

        void DisplayGBufferStateMessage(VAOEffectCommandBuffer vaoScript)
        {

            if (!vaoScript.UseGBuffer && vaoScript.ShouldUseGBuffer())
            {
                string reason = "";

                if (vaoScript.VaoCameraEvent != VAOEffectCommandBuffer.VAOCameraEventType.BeforeImageEffectsOpaque)
                {
                    reason = " Command buffer integration is different than BeforeImageEffectsOpaque.";
                }

#if UNITY_2017_1_OR_NEWER
                if (camera != null && camera.stereoEnabled
                    && isCameraSPSR(camera)
                    && camera.actualRenderingPath == RenderingPath.DeferredShading)
                {
                    reason = " You are running in single pass stereo mode which requires G-Buffer inputs.";
                }
#endif
                EditorGUILayout.HelpBox("Cannot turn G-Buffer depth&normals off, because current configuration requires it to be enabled." + reason, MessageType.Warning);
            }

        }

        private bool isCameraSPSR(Camera camera)
        {
            if (camera == null) return false;

#if UNITY_5_5_OR_NEWER
            if (camera.stereoEnabled)
            {
#if UNITY_2017_2_OR_NEWER

                return (UnityEngine.XR.XRSettings.eyeTextureDesc.vrUsage == VRTextureUsage.TwoEyes);
#else

#if !UNITY_WEBGL
                if (camera.stereoEnabled && PlayerSettings.stereoRenderingPath == StereoRenderingPath.SinglePass)
                    return true;
#endif

#endif
            }
#endif

            return false;
        }

        #region Custom header styles

        // Custom header foldout styles
        private static GUIStyle _customFoldoutStyle;
        private static GUIStyle customFoldoutStyle
        {
            get
            {
                if (_customFoldoutStyle == null)
                {
                    _customFoldoutStyle = new GUIStyle(EditorStyles.foldout)
                    {
                        fixedWidth = 12.0f
                    };
                }

                return _customFoldoutStyle;
            }
        }

        private static GUIStyle _customFoldinStyle;
        private static GUIStyle customFoldinStyle
        {
            get
            {
                if (_customFoldinStyle == null)
                {
                    _customFoldinStyle = new GUIStyle(EditorStyles.foldout)
                    {
                        fixedWidth = 12.0f
                    };

                    _customFoldinStyle.normal = _customFoldinStyle.onNormal;
                }

                return _customFoldinStyle;
            }
        }

        private static GUIStyle _customToggleStyle;
        private static GUIStyle customToggleStyle
        {
            get
            {
                if (_customToggleStyle == null)
                {
                    _customToggleStyle = new GUIStyle(EditorStyles.toggle)
                    {
                        fixedWidth = 12.0f
                    };
                }

                return _customToggleStyle;
            }
        }

        private static GUIStyle _customIconStyle;
        private static GUIStyle customIconStyle
        {
            get
            {
                if (_customIconStyle == null)
                {
                    _customIconStyle = new GUIStyle(EditorStyles.label)
                    {
                        fixedWidth = 18.0f,
                        fixedHeight = 18.0f
                    };
                }

                return _customIconStyle;
            }
        }


        private static GUIStyle _customLabelStyle;
        private static GUIStyle customLabelStyle
        {
            get
            {
                if (_customLabelStyle == null)
                {
                    _customLabelStyle = new GUIStyle(EditorStyles.label)
                    {
                        fontStyle = FontStyle.Normal
                    };
                }

                return _customLabelStyle;
            }
        }

        #endregion

        private static void HeaderWithToggle(string label, ref bool isFoldout, ref bool isEnabled, Texture icon = null, bool clickableIcon = true)
        {

            EditorGUILayout.BeginHorizontal();
            isFoldout = GUILayout.Toggle(isFoldout, "", isFoldout ? customFoldinStyle : customFoldoutStyle);
            isEnabled = GUILayout.Toggle(isEnabled, "", customToggleStyle);

            if (icon != null)
            {
                if (clickableIcon)
                    isFoldout = GUILayout.Toggle(isFoldout, icon, customLabelStyle);
                else
                    GUILayout.Label(icon);

                GUILayout.Space(-5.0f);
            }

            isFoldout = GUILayout.Toggle(isFoldout, label, customLabelStyle);
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
        }

    }

    #region Graph Widget

    public class GraphWidgetLine
    {
        public Vector3 From { get; set; }
        public Vector3 To { get; set; }
        public Color Color { get; set; }
        public float Thickness { get; set; }
    }

    public class GraphWidgetDrawingParameters
    {

        public IList<GraphWidgetLine> Lines { get; set; }

        /// <summary>
        /// Number of line segments that will be used to approximate function shape
        /// </summary>
        public uint GraphSegmentsCount { get; set; }

        /// <summary>
        /// Function to draw (X -> Y) 
        /// </summary>
        public Func<float, float> GraphFunction { get; set; }

        public Color GraphColor { get; set; }
        public float GraphThickness { get; set; }

        public float YScale { get; internal set; }
        public float MinY { get; internal set; }

        public int GridLinesXCount { get; set; }
        public float MaxFx { get; internal set; }

        public string LabelText { get; set; }
    }

    public class GraphWidget
    {
        private Vector3[] transformedLinePoints = new Vector3[2];
        private Vector3[] graphPoints;

        void TransformToRect(Rect rect, ref Vector3 v)
        {
            v.x = Mathf.Lerp(rect.x, rect.xMax, v.x);
            v.y = Mathf.Lerp(rect.yMax, rect.y, v.y);
        }

        private void DrawLine(Rect rect, float x1, float y1, float x2, float y2, Color color)
        {
            transformedLinePoints[0].x = x1;
            transformedLinePoints[0].y = y1;
            transformedLinePoints[1].x = x2;
            transformedLinePoints[1].y = y2;

            TransformToRect(rect, ref transformedLinePoints[0]);
            TransformToRect(rect, ref transformedLinePoints[1]);

            Handles.color = color;
            Handles.DrawPolyLine(transformedLinePoints);
        }

        private void DrawAALine(Rect rect, float thickness, float x1, float y1, float x2, float y2, Color color)
        {
            transformedLinePoints[0].x = x1;
            transformedLinePoints[0].y = y1;
            transformedLinePoints[1].x = x2;
            transformedLinePoints[1].y = y2;

            TransformToRect(rect, ref transformedLinePoints[0]);
            TransformToRect(rect, ref transformedLinePoints[1]);

            Handles.color = color;
            Handles.DrawPolyLine(transformedLinePoints);
        }

        public void Draw(GraphWidgetDrawingParameters drawingParameters)
        {
            Handles.color = Color.white; //< Reset to white to avoid Unity bugs

            Rect bgRect = GUILayoutUtility.GetRect(128, 70);
            Handles.DrawSolidRectangleWithOutline(bgRect, Color.grey, Color.black);

            // Draw grid lines
            Color gridColor = Color.black * 0.1f;
            DrawLine(bgRect, 0.0f, drawingParameters.MinY + drawingParameters.YScale,
                             1.0f, drawingParameters.MinY + drawingParameters.YScale, gridColor);

            DrawLine(bgRect, 0.0f, drawingParameters.MinY,
                             1.0f, drawingParameters.MinY, gridColor);

            float gridXStep = 1.0f / (drawingParameters.GridLinesXCount + 1);
            float gridX = gridXStep;
            for (int i = 0; i < drawingParameters.GridLinesXCount; i++)
            {
                DrawLine(bgRect, gridX, 0.0f,
                                 gridX, 1.0f, gridColor);

                gridX += gridXStep;
            }

            if (drawingParameters.GraphSegmentsCount > 0)
            {
                if (graphPoints == null || graphPoints.Length < drawingParameters.GraphSegmentsCount + 1)
                    graphPoints = new Vector3[drawingParameters.GraphSegmentsCount + 1];

                float x = 0.0f;
                float xStep = 1.0f / drawingParameters.GraphSegmentsCount;

                for (int i = 0; i < drawingParameters.GraphSegmentsCount + 1; i++)
                {
                    float y = drawingParameters.GraphFunction(x * drawingParameters.MaxFx);

                    y *= drawingParameters.YScale;
                    y += drawingParameters.MinY;

                    graphPoints[i].x = x;
                    graphPoints[i].y = y;
                    TransformToRect(bgRect, ref graphPoints[i]);
                    x += xStep;
                }

                Handles.color = drawingParameters.GraphColor;
                Handles.DrawAAPolyLine(drawingParameters.GraphThickness, graphPoints);
            }

            if (drawingParameters != null && drawingParameters.Lines != null)
            {
                foreach (var line in drawingParameters.Lines)
                {
                    DrawAALine(bgRect, line.Thickness, line.From.x, line.From.y, line.To.x, line.To.y, line.Color);
                }
            }

            // Label
            Vector3 labelPosition = new Vector3(0.01f, 0.99f);
            TransformToRect(bgRect, ref labelPosition);
            Handles.Label(labelPosition, drawingParameters.LabelText, EditorStyles.miniLabel);

        }

    }

    #endregion

#endif
}
