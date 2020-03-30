using UnityEngine;
using UnityEngine.Rendering;
using System.Linq;

[ExecuteInEditMode, AddComponentMenu("Image Effects/HBAO Integrated")]
[RequireComponent(typeof(Camera))]
public class HBAO_Integrated : HBAO_Core
{
    private CommandBuffer _hbaoCommandBuffer;
    private IntegrationStage _integrationStage;
    private Resolution _resolution;
    private DisplayMode _displayMode;
    private RenderingPath _renderingPath;
    private bool _hdr;
    private int _width;
    private int _height;
    private Quality _aoQuality;
    private Deinterleaving _deinterleaving;
    private bool _useMultiBounce;
    private bool _colorBleedingEnabled;
    private Blur _blurAmount;
    private bool _prepareInitialCommandBuffer;

    protected override void OnEnable()
    {
        base.OnEnable();

        if (_hbaoCommandBuffer == null)
        {
            _hbaoCommandBuffer = new CommandBuffer();
            _hbaoCommandBuffer.name = "HBAO";
        }

        _prepareInitialCommandBuffer = true;
    }

    protected override void OnDisable()
    {
        ClearCommandBuffer();

        base.OnDisable();
    }

    protected override void CheckParameters()
    {
        base.CheckParameters();

        var cameraEvent = GetCameraEvent();
        if (cameraEvent != CameraEvent.BeforeImageEffectsOpaque && !IsDeferredShading())
        {
            GeneralSettings settings = generalSettings;
            settings.integrationStage = IntegrationStage.BeforeImageEffectsOpaque;
            generalSettings = settings;
        }
        if (cameraEvent == CameraEvent.BeforeImageEffectsOpaque && aoSettings.perPixelNormals == PerPixelNormals.GBuffer)
        {
            AOSettings settings = aoSettings;
            settings.perPixelNormals = PerPixelNormals.Camera;
            aoSettings = settings;
        }
        else if (cameraEvent != CameraEvent.BeforeImageEffectsOpaque && aoSettings.perPixelNormals == PerPixelNormals.Camera)
        {
            AOSettings settings = aoSettings;
            settings.perPixelNormals = PerPixelNormals.GBuffer;
            aoSettings = settings;
        }
    }

    void OnPreRender()
    {
        if (hbaoShader == null || _hbaoCamera == null)
        {
            return;
        }

        _hbaoCamera.depthTextureMode |= DepthTextureMode.Depth;
        if (aoSettings.perPixelNormals == PerPixelNormals.Camera)
            _hbaoCamera.depthTextureMode |= DepthTextureMode.DepthNormals;

        CheckParameters();
        UpdateShaderProperties(2);
        UpdateShaderKeywords();

        bool prepareCommandBuffer = false;
        if (_integrationStage != generalSettings.integrationStage || _resolution != generalSettings.resolution || _displayMode != generalSettings.displayMode ||
            _renderingPath != _renderTarget.renderingPath || _hdr != _renderTarget.hdr || _width != _renderTarget.fullWidth || _height != _renderTarget.fullHeight ||
            _aoQuality != generalSettings.quality || _deinterleaving != generalSettings.deinterleaving || _useMultiBounce != aoSettings.useMultiBounce ||
            _colorBleedingEnabled != colorBleedingSettings.enabled || _blurAmount != blurSettings.amount)
        {
            _integrationStage = generalSettings.integrationStage;
            _resolution = generalSettings.resolution;
            _displayMode = generalSettings.displayMode;
            _renderingPath = _renderTarget.renderingPath;
            _hdr = _renderTarget.hdr;
            _width = _renderTarget.fullWidth;
            _height = _renderTarget.fullHeight;
            _aoQuality = generalSettings.quality;
            _deinterleaving = generalSettings.deinterleaving;
            _useMultiBounce = aoSettings.useMultiBounce;
            _colorBleedingEnabled = colorBleedingSettings.enabled;
            _blurAmount = blurSettings.amount;

            prepareCommandBuffer = true;
            //Debug.Log("Should prepare command buffer!");
        }

        if (prepareCommandBuffer || _prepareInitialCommandBuffer)
        {
            ClearCommandBuffer();

            var cameraEvent = GetCameraEvent();
            if (generalSettings.deinterleaving == Deinterleaving._2x)
                PrepareCommandBufferHBAODeinterleaved2x(cameraEvent);
            else if (generalSettings.deinterleaving == Deinterleaving._4x)
                PrepareCommandBufferHBAODeinterleaved4x(cameraEvent);
            else
                PrepareCommandBufferHBAO(cameraEvent);

            _hbaoCamera.AddCommandBuffer(cameraEvent, _hbaoCommandBuffer);

            _prepareInitialCommandBuffer = false;
        }
    }

    private void PrepareCommandBufferHBAO(CameraEvent cameraEvent)
    {
        var mainTexId = new RenderTargetIdentifier(ShaderProperties.mainTex);
        var hbaoTexId = new RenderTargetIdentifier(ShaderProperties.hbaoTex);

        _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.hbaoTex, _renderTarget.fullWidth / _renderTarget.downsamplingFactor, _renderTarget.fullHeight / _renderTarget.downsamplingFactor);

        _hbaoCommandBuffer.SetRenderTarget(hbaoTexId);
        _hbaoCommandBuffer.ClearRenderTarget(false, true, Color.white);

        _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, hbaoTexId, _hbaoMaterial, GetAoPass()); // hbao

        if (blurSettings.amount != Blur.None)
        {
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, (_renderTarget.fullWidth / _renderTarget.downsamplingFactor) / _renderTarget.blurDownsamplingFactor,
                                                                        (_renderTarget.fullHeight / _renderTarget.downsamplingFactor) / _renderTarget.blurDownsamplingFactor);
            _hbaoCommandBuffer.Blit(hbaoTexId, mainTexId, _hbaoMaterial, GetBlurXPass()); // blur X
            _hbaoCommandBuffer.Blit(mainTexId, hbaoTexId, _hbaoMaterial, GetBlurYPass()); // blur Y

            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);
        }

        _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.hbaoTex, hbaoTexId);
        RenderHBAO(cameraEvent);

        _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.hbaoTex);
    }

    private void PrepareCommandBufferHBAODeinterleaved2x(CameraEvent cameraEvent)
    {
        var mainTexId = new RenderTargetIdentifier(ShaderProperties.mainTex);
        var hbaoTexId = new RenderTargetIdentifier(ShaderProperties.hbaoTex);
        var mrtDepthId = new RenderTargetIdentifier[] { ShaderProperties.mrtDepthTex[0], ShaderProperties.mrtDepthTex[1], ShaderProperties.mrtDepthTex[2], ShaderProperties.mrtDepthTex[3] };
        var mrtNormId = new RenderTargetIdentifier[] { ShaderProperties.mrtNrmTex[0], ShaderProperties.mrtNrmTex[1], ShaderProperties.mrtNrmTex[2], ShaderProperties.mrtNrmTex[3] };
        var mrtHBAOId = new RenderTargetIdentifier[] { ShaderProperties.mrtHBAOTex[0], ShaderProperties.mrtHBAOTex[1], ShaderProperties.mrtHBAOTex[2], ShaderProperties.mrtHBAOTex[3] };

        // initialize render textures & buffers
        for (int i = 0; i < NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mrtDepthTex[i], _renderTarget.layerWidth, _renderTarget.layerHeight, 0, FilterMode.Point, RenderTextureFormat.RFloat);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mrtNrmTex[i], _renderTarget.layerWidth, _renderTarget.layerHeight, 0, FilterMode.Point, RenderTextureFormat.ARGB2101010);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mrtHBAOTex[i], _renderTarget.layerWidth, _renderTarget.layerHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
        }

        // deinterleave depth & normals 2x2
        _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[0], new Vector2(0, 0));
        _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[1], new Vector2(1, 0));
        _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[2], new Vector2(0, 1));
        _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[3], new Vector2(1, 1));
        _hbaoCommandBuffer.SetRenderTarget(mrtDepthId, mrtDepthId[0]);
        _hbaoCommandBuffer.DrawMesh(quadMesh, Matrix4x4.identity, _hbaoMaterial, 0, Pass.Depth_Deinterleaving_2x2); // outputs 4 render textures
        _hbaoCommandBuffer.SetRenderTarget(mrtNormId, mrtNormId[0]);
        _hbaoCommandBuffer.DrawMesh(quadMesh, Matrix4x4.identity, _hbaoMaterial, 0, Pass.Normals_Deinterleaving_2x2); // outputs 4 render textures

        // calculate AO on each layer
        for (int i = 0; i < NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.depthTex, mrtDepthId[i]);
            _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.normalsTex, mrtNormId[i]);
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.jitter, _jitter[i]);
            _hbaoCommandBuffer.SetRenderTarget(mrtHBAOId[i]);
            _hbaoCommandBuffer.ClearRenderTarget(false, true, Color.white);
            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, mrtHBAOId[i], _hbaoMaterial, GetAoDeinterleavedPass()); // hbao
        }

        // build atlas
        _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, _renderTarget.fullWidth, _renderTarget.fullHeight);
        for (int i = 0; i < NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.layerOffset, new Vector2((i & 1) * _renderTarget.layerWidth, (i >> 1) * _renderTarget.layerHeight));
            _hbaoCommandBuffer.Blit(mrtHBAOId[i], mainTexId, _hbaoMaterial, Pass.Atlas);
        }

        // reinterleave
        _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.hbaoTex, _renderTarget.fullWidth, _renderTarget.fullHeight);
        _hbaoCommandBuffer.Blit(mainTexId, hbaoTexId, _hbaoMaterial, Pass.Reinterleaving_2x2);

        if (blurSettings.amount != Blur.None)
        {
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, _renderTarget.fullWidth / _renderTarget.blurDownsamplingFactor,
                                                                        _renderTarget.fullHeight / _renderTarget.blurDownsamplingFactor);
            _hbaoCommandBuffer.Blit(hbaoTexId, mainTexId, _hbaoMaterial, GetBlurXPass()); // blur X
            _hbaoCommandBuffer.Blit(mainTexId, hbaoTexId, _hbaoMaterial, GetBlurYPass()); // blur Y
        }

        _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);

        for (int i = 0; i < NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mrtHBAOTex[i]);
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mrtNrmTex[i]);
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mrtDepthTex[i]);
        }

        _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.hbaoTex, hbaoTexId);
        RenderHBAO(cameraEvent);

        _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.hbaoTex);
    }

    private void PrepareCommandBufferHBAODeinterleaved4x(CameraEvent cameraEvent)
    {
        var mainTexId = new RenderTargetIdentifier(ShaderProperties.mainTex);
        var hbaoTexId = new RenderTargetIdentifier(ShaderProperties.hbaoTex);
        var mrtDepthId = new RenderTargetIdentifier[4 * NUM_MRTS];
        var mrtNormId = new RenderTargetIdentifier[4 * NUM_MRTS];
        var mrtHBAOId = new RenderTargetIdentifier[4 * NUM_MRTS];

        // initialize render textures & buffers
        for (int i = 0; i < 4 * NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mrtDepthTex[i], _renderTarget.layerWidth, _renderTarget.layerHeight, 0, FilterMode.Point, RenderTextureFormat.RFloat);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mrtNrmTex[i], _renderTarget.layerWidth, _renderTarget.layerHeight, 0, FilterMode.Point, RenderTextureFormat.ARGB2101010);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mrtHBAOTex[i], _renderTarget.layerWidth, _renderTarget.layerHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

            mrtDepthId[i] = ShaderProperties.mrtDepthTex[i];
            mrtNormId[i] = ShaderProperties.mrtNrmTex[i];
            mrtHBAOId[i] = ShaderProperties.mrtHBAOTex[i];
        }

        // deinterleave depth & normals 4x4
        for (int i = 0; i < NUM_MRTS; i++)
        {
            int offsetX = (i & 1) << 1;
            int offsetY = (i >> 1) << 1;

            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[0], new Vector2(offsetX + 0, offsetY + 0));
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[1], new Vector2(offsetX + 1, offsetY + 0));
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[2], new Vector2(offsetX + 0, offsetY + 1));
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.deinterleavingOffset[3], new Vector2(offsetX + 1, offsetY + 1));
            var mrtDepth = new RenderTargetIdentifier[] { mrtDepthId[(i << 2) + 0], mrtDepthId[(i << 2) + 1], mrtDepthId[(i << 2) + 2], mrtDepthId[(i << 2) + 3] };
            var mrtNorm = new RenderTargetIdentifier[] { mrtNormId[(i << 2) + 0], mrtNormId[(i << 2) + 1], mrtNormId[(i << 2) + 2], mrtNormId[(i << 2) + 3] };
            _hbaoCommandBuffer.SetRenderTarget(mrtDepth, mrtDepth[0]);
            _hbaoCommandBuffer.DrawMesh(quadMesh, Matrix4x4.identity, _hbaoMaterial, 0, Pass.Depth_Deinterleaving_4x4); // outputs 4 render textures
            _hbaoCommandBuffer.SetRenderTarget(mrtNorm, mrtNorm[0]);
            _hbaoCommandBuffer.DrawMesh(quadMesh, Matrix4x4.identity, _hbaoMaterial, 0, Pass.Normals_Deinterleaving_4x4); // outputs 4 render textures
        }

        // calculate AO on each layer
        for (int i = 0; i < 4 * NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.depthTex, mrtDepthId[i]);
            _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.normalsTex, mrtNormId[i]);
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.jitter, _jitter[i]);
            _hbaoCommandBuffer.SetRenderTarget(mrtHBAOId[i]);
            _hbaoCommandBuffer.ClearRenderTarget(false, true, Color.white);
            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, mrtHBAOId[i], _hbaoMaterial, GetAoDeinterleavedPass()); // hbao
        }

        // build atlas
        _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, _renderTarget.fullWidth, _renderTarget.fullHeight);
        for (int i = 0; i < 4 * NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.SetGlobalVector(ShaderProperties.layerOffset, new Vector2(((i & 1) + (((i & 7) >> 2) << 1)) * _renderTarget.layerWidth,
                                                                                        (((i & 3) >> 1) + ((i >> 3) << 1)) * _renderTarget.layerHeight));
            _hbaoCommandBuffer.Blit(mrtHBAOId[i], mainTexId, _hbaoMaterial, Pass.Atlas);
        }

        // reinterleave
        _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.hbaoTex, _renderTarget.fullWidth, _renderTarget.fullHeight);
        _hbaoCommandBuffer.Blit(mainTexId, hbaoTexId, _hbaoMaterial, Pass.Reinterleaving_4x4);

        if (blurSettings.amount != Blur.None)
        {
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, _renderTarget.fullWidth / _renderTarget.blurDownsamplingFactor,
                                                                        _renderTarget.fullHeight / _renderTarget.blurDownsamplingFactor);
            _hbaoCommandBuffer.Blit(hbaoTexId, mainTexId, _hbaoMaterial, GetBlurXPass()); // blur X
            _hbaoCommandBuffer.Blit(mainTexId, hbaoTexId, _hbaoMaterial, GetBlurYPass()); // blur Y
        }

        _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);

        for (int i = 0; i < 4 * NUM_MRTS; i++)
        {
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mrtHBAOTex[i]);
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mrtNrmTex[i]);
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mrtDepthTex[i]);
        }

        _hbaoCommandBuffer.SetGlobalTexture(ShaderProperties.hbaoTex, hbaoTexId);
        RenderHBAO(cameraEvent);

        _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.hbaoTex);
    }

    private void RenderHBAO(CameraEvent cameraEvent)
    {
        //_hbaoCommandBuffer.SetGlobalVector("_FullRes_TexelSize", new Vector4(1.0f / renderTarget.width, 1.0f / renderTarget.height, renderTarget.width, renderTarget.height));

        if (generalSettings.displayMode == DisplayMode.Normal)
        {
#if !(UNITY_5_1 || UNITY_5_0)
            if (cameraEvent == CameraEvent.BeforeReflections)
            {
                var mrt = new RenderTargetIdentifier[] {
                    BuiltinRenderTextureType.GBuffer0, // Albedo, Occ
                    _renderTarget.hdr ? BuiltinRenderTextureType.CameraTarget : BuiltinRenderTextureType.GBuffer3 // Ambient
                };

                if (_renderTarget.hdr)
                {
                    var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                    _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);

                    _hbaoCommandBuffer.SetRenderTarget(mrt, BuiltinRenderTextureType.CameraTarget);
                    _hbaoCommandBuffer.DrawMesh(quadMesh, Matrix4x4.identity, _hbaoMaterial, 0, Pass.Combine_Deffered_Multiplicative); // AO blend pass (multiplicative)
                    if (colorBleedingSettings.enabled)
                        _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial, Pass.Combine_ColorBleeding); // Color Bleeding blend pass (additive)

                    _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
                }
                else
                {
                    var rt0TexId = new RenderTargetIdentifier(ShaderProperties.rt0Tex);
                    var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                    _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt0Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGB32);
                    _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGB2101010);
                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.GBuffer0, rt0TexId);
                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.GBuffer3, rt3TexId);

                    _hbaoCommandBuffer.SetRenderTarget(mrt, BuiltinRenderTextureType.GBuffer3);
                    _hbaoCommandBuffer.DrawMesh(quadMesh, Matrix4x4.identity, _hbaoMaterial, 0, Pass.Combine_Deffered); // final pass

                    _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
                    _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt0Tex);
                }
            }
#endif

            if (cameraEvent == CameraEvent.AfterLighting)
            {
                if (_renderTarget.hdr)
                {
                    if (_useMultiBounce)
                    {
                        var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                        _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

                        _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);
                    }

                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial, 
                                            _useMultiBounce ? Pass.Combine_Integrated_Multiplicative_MultiBounce : Pass.Combine_Integrated_Multiplicative); // AO blend pass (multiplicative)
                    if (colorBleedingSettings.enabled)
                        _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial, Pass.Combine_ColorBleeding); // Color Bleeding blend pass (additive)

                    if (_useMultiBounce)
                        _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
                }
                else
                {
                    var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                    _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.GBuffer3, rt3TexId);

                    _hbaoCommandBuffer.Blit(rt3TexId, BuiltinRenderTextureType.GBuffer3, _hbaoMaterial, _useMultiBounce ? Pass.Combine_Integrated_MultiBounce : Pass.Combine_Integrated); // final pass

                    _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
                }
            }
            else if (cameraEvent == CameraEvent.BeforeImageEffectsOpaque)
            {
                if (_useMultiBounce)
                {
                    var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                    _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);

                    _hbaoCommandBuffer.Blit(rt3TexId, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial, Pass.Combine_Integrated_Multiplicative_MultiBounce); // AO blend pass (multiplicative)
                }
                else
                {
                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial, Pass.Combine_Integrated_Multiplicative); // AO blend pass (multiplicative)
                }
                if (colorBleedingSettings.enabled)
                    _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial, Pass.Combine_ColorBleeding); // Color Bleeding blend pass (additive)

                if (_useMultiBounce)
                    _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
            }
        }
        else if (generalSettings.displayMode == DisplayMode.AOOnly)
        {
            if (_useMultiBounce)
            {
                var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);
            }

            var mainTexId = new RenderTargetIdentifier(ShaderProperties.mainTex);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, _renderTarget.width, _renderTarget.height);
            _hbaoCommandBuffer.SetRenderTarget(mainTexId);
            _hbaoCommandBuffer.ClearRenderTarget(false, true, Color.white);
            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, mainTexId, _hbaoMaterial,
                                    _useMultiBounce ? Pass.Combine_Integrated_Multiplicative_MultiBounce : Pass.Combine_Integrated_Multiplicative); // AO blend pass (multiplicative)
            _hbaoCommandBuffer.Blit(mainTexId, BuiltinRenderTextureType.CameraTarget);
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);

            if (_useMultiBounce)
                _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
        }
        else if (generalSettings.displayMode == DisplayMode.ColorBleedingOnly)
        {
            var mainTexId = new RenderTargetIdentifier(ShaderProperties.mainTex);
            _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.mainTex, _renderTarget.width, _renderTarget.height);
            _hbaoCommandBuffer.SetRenderTarget(mainTexId);
            _hbaoCommandBuffer.ClearRenderTarget(false, true, Color.black);
            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, mainTexId, _hbaoMaterial, Pass.Combine_ColorBleeding); // Color Bleeding blend pass (additive)
            _hbaoCommandBuffer.Blit(mainTexId, BuiltinRenderTextureType.CameraTarget);
            _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.mainTex);

        }
        else if (generalSettings.displayMode == DisplayMode.SplitWithAOAndAOOnly)
        {
            if (_useMultiBounce)
            {
                var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);
            }

            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial,
                                    _useMultiBounce ? Pass.Combine_Integrated_Multiplicative_MultiBounce : Pass.Combine_Integrated_Multiplicative); // AO blend pass (multiplicative)
            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial,
                                    _useMultiBounce ? Pass.Debug_Split_Additive_MultiBounce : Pass.Debug_Split_Additive); // AO Only blend pass (additive)

            if (_useMultiBounce)
                _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
        }
        else if (generalSettings.displayMode == DisplayMode.SplitWithoutAOAndAOOnly)
        {
            if (_useMultiBounce)
            {
                var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);
            }

            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial,
                                    _useMultiBounce ? Pass.Debug_Split_Additive_MultiBounce : Pass.Debug_Split_Additive); // AO Only blend pass (additive)

            if (_useMultiBounce)
                _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
        }
        else if (generalSettings.displayMode == DisplayMode.SplitWithoutAOAndWithAO)
        {
            if (_useMultiBounce)
            {
                var rt3TexId = new RenderTargetIdentifier(ShaderProperties.rt3Tex);
                _hbaoCommandBuffer.GetTemporaryRT(ShaderProperties.rt3Tex, _renderTarget.fullWidth, _renderTarget.fullHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, rt3TexId);
            }

            _hbaoCommandBuffer.Blit(BuiltinRenderTextureType.CameraTarget, BuiltinRenderTextureType.CameraTarget, _hbaoMaterial,
                                    _useMultiBounce ? Pass.Debug_Split_Multiplicative_MultiBounce : Pass.Debug_Split_Multiplicative); // AO Only blend pass (multiplicative)

            if (_useMultiBounce)
                _hbaoCommandBuffer.ReleaseTemporaryRT(ShaderProperties.rt3Tex);
        }
    }

    private void ClearCommandBuffer()
    {
        if (_hbaoCommandBuffer != null)
        {
            if (_hbaoCamera != null)
            {
                _hbaoCamera.RemoveCommandBuffer(CameraEvent.BeforeImageEffectsOpaque, _hbaoCommandBuffer);
                _hbaoCamera.RemoveCommandBuffer(CameraEvent.AfterLighting, _hbaoCommandBuffer);
#if !(UNITY_5_1 || UNITY_5_0)
                _hbaoCamera.RemoveCommandBuffer(CameraEvent.BeforeReflections, _hbaoCommandBuffer);
#endif
            }
            _hbaoCommandBuffer.Clear();
        }
    }

    private CameraEvent GetCameraEvent()
    {
        if (generalSettings.displayMode != DisplayMode.Normal)
            return CameraEvent.BeforeImageEffectsOpaque;

        switch (generalSettings.integrationStage)
        {
#if !(UNITY_5_1 || UNITY_5_0)
            case IntegrationStage.BeforeReflections:
                return CameraEvent.BeforeReflections;
#endif
            case IntegrationStage.AfterLighting:
                return CameraEvent.AfterLighting;
            case IntegrationStage.BeforeImageEffectsOpaque:
            default:
                return CameraEvent.BeforeImageEffectsOpaque;
        }
    }
}
