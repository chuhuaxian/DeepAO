// Copyright (c) 2016-2018 Jakub Boksansky - All Rights Reserved
// Volumetric Ambient Occlusion Unity Plugin 2.0

#include "UnityCG.cginc"
		
// ========================================================================
// Uniform definitions 
// ========================================================================

// Unity built-ins
sampler2D _MainTex;
float4 _MainTex_TexelSize;
float4 _ProjInfo;
float4 _MainTex_ST;

sampler2D_half _CameraMotionVectorsTexture;
sampler2D _CameraDepthNormalsTexture;
float4 _CameraDepthNormalsTexture_ST;
sampler2D _CameraGBufferTexture2;
sampler2D_float _CameraDepthTexture;

// Culling pre-pass
sampler2D cullingPrepassTexture;
uniform int cullingPrepassMode;
uniform float2 cullingPrepassTexelSize;

// Projection matrices
uniform float4x4 invProjMatrix;
uniform float4 screenProjection;
uniform float4 screenUnprojection;
uniform float4 screenUnprojection2;

// Radius setting
uniform float halfRadius;
uniform float halfRadiusSquared;
uniform float radius;

uniform float halfRadiusWeight;
uniform float quarterRadiusWeight;

// Sample set
uniform int sampleCount;
uniform int fourSamplesStartIndex;
uniform int eightSamplesStartIndex;
uniform float4 samples[63];
uniform int samplesStartIndex;

// Temporal AO samples
sampler2D temporalSamples;

// AO appearance
uniform float aoPower;
uniform float aoPresence;
uniform float aoThickness;
uniform float bordersIntensity;
uniform float4 colorTint; 
		
// AO radius limits
uniform int maxRadiusEnabled;
uniform float maxRadiusCutoffDepth;
		
uniform int minRadiusEnabled;
uniform float minRadiusCutoffDepth;
uniform float minRadiusSoftness;

uniform float subpixelRadiusCutoffDepth;

// GI appearance
uniform float giPower;
uniform float giPresence;
uniform float giSameHueAttenuationBrightness;
uniform int giQuality;
uniform int giBackfaces;
uniform	int giBlur;
uniform int giSelfOcclusionFix;

uniform int giSameHueAttenuationEnabled;
uniform float giSameHueAttenuationThreshold;
uniform float giSameHueAttenuationWidth;
uniform float giSameHueAttenuationSaturationThreshold;
uniform float giSameHueAttenuationSaturationWidth;

// Adaptive sampling settings
uniform float adaptiveMin;
uniform float adaptiveMax;
uniform int adaptiveMode;
uniform float quarterResBufferMaxDistance;
uniform float halfResBufferMaxDistance;
uniform int quarterRadiusSamplesOffset;
uniform int halfRadiusSamplesOffset;

uniform int maxSamplesCount;

// Hierarchical Buffers 
sampler2D depthNormalsTexture2;
sampler2D depthNormalsTexture4;

// Luma settings
uniform float LumaThreshold;
uniform float LumaKneeWidth;
uniform float LumaTwiceKneeWidthRcp;
uniform float LumaKneeLinearity;
uniform int LumaMode;
		
// Blur settings
uniform int enhancedBlurSize;
uniform float4 gauss[99];
uniform float gaussWeight;
uniform float blurDepthThreshold;

// Utilities
sampler2D textureAO;
uniform float2 texelSize;
uniform float cameraFarPlane;
uniform int UseCameraFarPlane;
uniform float projMatrix11;
uniform float2 inputTexDimensions;
uniform int noVertexTransform;
uniform int historyReady;
uniform float maxRadiusOnScreen;
sampler2D noiseTexture;
sampler2D aoHistoryTexture;
sampler2D giHistoryTexture;
sampler2D gi2HistoryTexture;
sampler2D gi3HistoryTexture;
sampler2D cbInputTex;
uniform int enableReprojection;
uniform int useUnityMotionVectors;
uniform float2 noiseTexelSizeRcp;
uniform float4 temporalTexelSizeRcp;
uniform float2 texelSizeRcp;
uniform int frameNumber;
uniform float ssaoBias;
uniform int flipY;
uniform int useGBuffer;
uniform int hierarchicalBufferEnabled;
uniform int hwBlendingEnabled;
uniform int useLogEmissiveBuffer;
uniform int useLogBufferInput;
uniform int outputAOOnly;
uniform int useFastBlur;
uniform int isLumaSensitive;
uniform int useDedicatedDepthBuffer;
sampler2D emissionTexture;		
sampler2D occlusionTexture;		
uniform int isImageEffectMode;
uniform int normalsSource;
uniform int useSPSRFriendlyTransform;
sampler2D texCopySource;
sampler2D temporalTexCopySource;
uniform float4x4 motionVectorsMatrix;
uniform float4x4 motionVectorsMatrix2;

// ========================================================================
// Structs definitions 
// ========================================================================

struct v2fShed {
	float4 pos : SV_POSITION;
	#ifdef UNITY_SINGLE_PASS_STEREO
	float4 shed2 : TEXCOORD2;
	#endif
	float4 shed : TEXCOORD1;
	float2 uv : TEXCOORD0;
};
		
struct v2fSingle {
	float4 pos : SV_POSITION;
	float2 uv : TEXCOORD0;
};

struct v2fDouble {
	float4 pos : SV_POSITION;
	float2 uv[2] : TEXCOORD0;
};

struct AoOutput
{
	half4 albedoAO : SV_Target0;
	half4 emissionLighting : SV_Target1;
};

#ifdef WFORCE_VAO_COLORBLEED_OFF

struct TemporalAoOutput
{
	half4 history : SV_Target0;
	half4 aoDepth : SV_Target1;
};

#else

struct TemporalAoOutput
{
	half4 history : SV_Target0;
	half4 aoDepth : SV_Target1;
	half4 gi : SV_Target2;
	half4 gi2 : SV_Target3;
	half4 gi3 : SV_Target4;
};

#endif

// ========================================================================
// Defines
// ========================================================================

// Uncomment to use optimized sqrt function by M. Drobot - WARNING: may not be supported on all platforms (e.g. WebGL)
//#define WFORCE_USE_FAST_SQRT

#if !defined(SHADER_API_D3D9) && !defined(SHADER_API_GLES) && !defined(SHADER_API_GLES3)
	#define WFORCE_CAN_UNROLL
#else
	#define WFORCE_CANNOT_UNROLL
#endif


#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)
	#define WFORCE_VAO_MAIN_PASS_RETURN_TYPE TemporalAoOutput
	#ifdef WFORCE_VAO_COLORBLEED_OFF
		#define WFORCE_VAO_WHITE half2(1.0f, 1.0f)
		#define WFORCE_VAO_BLACK half2(0.0f, 1.0f)
	#else
		#define WFORCE_VAO_WHITE half4(1.0f, 1.0f, 1.0f, 1.0f)
		#define WFORCE_VAO_BLACK half4(0.0f, 1.0f, 1.0f, 1.0f)
	#endif
#else

	#ifdef SHADER_API_D3D9
		#define WFORCE_VAO_MAIN_PASS_RETURN_TYPE half4
		#define WFORCE_VAO_WHITE half4(1.0f, 1.0f, 1.0f, 1.0f)
		#define WFORCE_VAO_BLACK half4(0.0f, 1.0f, 1.0f, 1.0f)
	#else
		#ifdef WFORCE_VAO_COLORBLEED_OFF
			#define WFORCE_VAO_MAIN_PASS_RETURN_TYPE half2
			#define WFORCE_VAO_WHITE half2(1.0f, 1.0f)
			#define WFORCE_VAO_BLACK half2(0.0f, 1.0f)
		#else
			#define WFORCE_VAO_MAIN_PASS_RETURN_TYPE half4
			#define WFORCE_VAO_WHITE half4(1.0f, 1.0f, 1.0f, 1.0f)
			#define WFORCE_VAO_BLACK half4(0.0f, 1.0f, 1.0f, 1.0f)
		#endif
	#endif
#endif
		
#ifndef SHADER_API_GLCORE
#ifndef SHADER_API_OPENGL
#ifndef SHADER_API_GLES
#ifndef SHADER_API_GLES3
#ifndef SHADER_API_VULKAN
	#define WFORCE_VAO_OPENGL_OFF
#endif
#endif
#endif
#endif
#endif
		
#if defined(SHADER_API_GLES)
#define WFORCE_UNROLL(n)
#else
#define WFORCE_UNROLL(n) [unroll(n)]
#endif

#define WFORCE_STANDARD_VAO 1
#define WFORCE_RAYCAST_AO 2

// ========================================================================
// Helper functions
// ========================================================================
		
float3 RGBToHSV(float3 rgb)
{
	// result.x = Hue			[0.0, 360.0] in degrees
	// result.y = Saturation	[0.0, 1.0]
	// result.z = Value			[0.0, 1.0]

	float3 result = float3(0.0f, 0.0f, 0.0f);

	float cMax = max(rgb.r, max(rgb.g, rgb.b));
	float cMin = min(rgb.r, min(rgb.g, rgb.b));

	float delta = cMax - cMin;

	if (cMax > 0.000001f) {
		result.y = delta / cMax;
					
		if (rgb.r == cMax) {
			result.x = (rgb.g - rgb.b) / delta;			// between yellow & magenta
		} else if (rgb.g == cMax) {
			result.x = 2.0f + (rgb.b - rgb.r) / delta;	// between cyan & yellow
		} else {
			result.x = 4.0f + (rgb.r - rgb.g) / delta;	// between magenta & cyan
		}

		result.x *= 60.0f;			

		if(result.x < 0.0f) result.x += 360.0f;

		result.z = cMax;

		return result;

	} else {
		// Undefined (grey - saturation is zero)
		return float3(0.0f, 0.0f, cMax);
	}

}
			
float3 fetchNormal(float2 uv) {

	float3 normal = mul((float3x3)unity_WorldToCamera, tex2Dlod(_CameraGBufferTexture2, float4(uv, 0.0f, 0.0f)).xyz * 2.0f - 1.0f);
	normal.z = -normal.z;

	return normal;
}

float fetchDepth(float2 uv, float farPlane) {
	if (useGBuffer == 0) {
		return -DecodeFloatRG(tex2Dlod(_CameraDepthNormalsTexture, float4(uv, 0.0f, 0.0f)).zw) * farPlane;
	} else {
		return -Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(uv, 0.0f, 0.0f)).r) * farPlane;
	}
}

void fetchDepthNormal(float2 uv, out float depth, out float3 normal) {
	if (useGBuffer == 0) {
		DecodeDepthNormal(tex2Dlod(_CameraDepthNormalsTexture, float4(uv, 0.0f, 0.0f)), depth, normal);

		if (useDedicatedDepthBuffer != 0) {
			depth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(uv, 0.0f, 0.0f)).r);
		}

	} else {
		depth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(uv, 0.0f, 0.0f)).r);
		normal = fetchNormal(uv);
	}
}
		
inline void decodeDepth(float4 enc, out float depth)
{
	depth = DecodeFloatRG(enc.zw);
}

float3 getViewSpacePositionUsingDepth(float2 uv, float pixelDepth) {

#ifdef UNITY_SINGLE_PASS_STEREO
	if (uv.x > 0.5f) {
		uv.x = uv.x * 2.0f - 1.0f;
	}
	else {
		uv.x = uv.x * 2.0f;
	}
#endif
	float4 screenPosition = float4(uv * 2.0f - 1.0f, 1.0f, 1.0f);
	float3 ray = (screenPosition.xyz * screenUnprojection.xyz + screenUnprojection2.xyz) * screenUnprojection2.w;
	return ray * pixelDepth;
}

float3 getViewSpacePosition(float2 uv) {

	float pixelDepth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(uv, 0.0f, 0.0f)).r);

	return getViewSpacePositionUsingDepth(uv, pixelDepth);
}

half3 calculateNormal(float2 uv) {

	float3 centerPosition = getViewSpacePosition(uv);

	float3 upPosition = getViewSpacePosition(uv + float2(0, texelSizeRcp.y));
	float3 rightPosition = getViewSpacePosition(uv + float2(texelSizeRcp.x, 0));
	float3 downPosition = getViewSpacePosition(uv + float2(0, -texelSizeRcp.y));
	float3 leftPosition = getViewSpacePosition(uv + float2(-texelSizeRcp.x, 0));

	float3 xSample1 = leftPosition - centerPosition;
	float3 xSample2 = centerPosition - rightPosition;
	float3 xSample = (length(xSample1) < length(xSample2)) ? xSample1 : xSample2;

	float3 ySample1 = upPosition - centerPosition;
	float3 ySample2 = centerPosition - downPosition;
	float3 ySample = (length(ySample1) < length(ySample2)) ? ySample1 : ySample2;

	return normalize(cross(ySample, xSample));
}

void fetchDepthCalculateNormal(float2 uv, out float depth, out float3 normal) {
	depth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(uv, 0.0f, 0.0f)).r);
	normal = calculateNormal(uv);
}

float getFarPlane(int useCameraFarPlane) {
	if (useCameraFarPlane != 0) {
		return cameraFarPlane;
	} else {
		return _ProjectionParams.z;
	}
}
		
float luma(float3 color) {
	return 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;
}

#ifdef WFORCE_USE_FAST_SQRT

	// Fast square root by Michal Drobot
	float fastSqrtNR0(float inX)
	{
		return asfloat(0x1FBD1DF5 + (asint(inX) >> 1));
	}

	// Fast square root by Michal Drobot
	float2 fastSqrtNR0(float2 inX)
	{
		return asfloat(0x1FBD1DF5 + (asint(inX) >> 1));
	}

#endif

float vaoSqrt(float x) {
#ifdef WFORCE_USE_FAST_SQRT
	return fastSqrtNR0(x);
#else
	return sqrt(x);
#endif
}

float2 vaoSqrt(float2 x) {
#ifdef WFORCE_USE_FAST_SQRT
	return fastSqrtNR0(x);
#else
	return sqrt(x);
#endif
}

float getRadiusForDepthAndScreenRadius(float pixelDepth, float maxRadiusOnScreen) {
	return -(pixelDepth * maxRadiusOnScreen) / projMatrix11;
}
		
#ifdef WFORCE_VAO_COLORBLEED_ON
void applyLuma(float3 mainColor, float accIn, out float acc, float3 giIn, out float3 gi) {
#else
void applyLuma(float3 mainColor, float accIn, out float acc) {
#endif	
			
	float Y;
	if (LumaMode == 1) {
		Y = luma(mainColor);
	} else {
		Y = max(max(mainColor.r, mainColor.g), mainColor.b);
	}

	Y = (Y - (LumaThreshold - LumaKneeWidth)) * LumaTwiceKneeWidthRcp;
	float x = min(1.0f, max(0.0f, Y));
	float n = ((-pow(x, LumaKneeLinearity) + 1));
	acc = lerp(1.0f, accIn, n);
				
	#ifdef WFORCE_VAO_COLORBLEED_ON
		gi = lerp(float3(1.0f, 1.0f, 1.0f), giIn, n);
	#endif
}
		
float3 processGi(float3 mainColor, float3 giIn, float ao) {
	float3 gi = giIn;

	float sMax = max(gi.r, max(gi.g, gi.b));
	float sMin = min(gi.r, min(gi.g, gi.b));
	float sat = 0.0f;
	if (sMax > 0.01f) sat = (sMax - sMin) / sMax;
	float _satMapped = 1.0f - (sat*giPresence);
	gi = lerp(float3(1.0f, 1.0f, 1.0f), gi, _satMapped);

	gi *= ao / max(max(gi.x, gi.y),gi.z); //< Tatry su krute

	// Sme hue attenuation 
	if (giSameHueAttenuationEnabled != 0) {

		float giHue = RGBToHSV(gi).x;
		float3 mainHSV = RGBToHSV(mainColor);
		float mainHue = mainHSV.x;
		float mainValue = mainHSV.y;
		float hueDiff = abs(giHue - mainHue);
		float3 giHueSuppressed = lerp(float3(1, 1, 1), gi, smoothstep(giSameHueAttenuationThreshold - giSameHueAttenuationWidth,  giSameHueAttenuationThreshold + giSameHueAttenuationWidth, hueDiff));
		gi = lerp(gi, giHueSuppressed, smoothstep(giSameHueAttenuationSaturationThreshold - giSameHueAttenuationSaturationWidth,  giSameHueAttenuationSaturationThreshold + giSameHueAttenuationSaturationWidth, mainValue));
		gi = lerp(gi, gi * (1.0f / max(max(gi.x, gi.y),gi.z)), giSameHueAttenuationBrightness);

	}

	return gi;
}

#ifdef WFORCE_VAO_COLORBLEED_OFF
half downscaleDepthNormals(v2fDouble input) {
#else
half4 downscaleDepthNormals(v2fDouble input) {
#endif

	#ifdef WFORCE_VAO_COLORBLEED_OFF

		float depth;

		if (useGBuffer == 0 && useDedicatedDepthBuffer == 0) {
			decodeDepth(tex2Dlod(_CameraDepthNormalsTexture, float4(input.uv[1], 0.0f, 0.0f)), depth);
		} else {
			depth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(input.uv[1], 0.0f, 0.0f)).r);
		}
				
		return depth;

	#else

		float depth;
		float3 normal;
				
		if (normalsSource == 1) {
			fetchDepthNormal(input.uv[1], depth, normal);
		}
		else {
			fetchDepthCalculateNormal(input.uv[1], depth, normal);
		}

		return EncodeDepthNormal(depth, normal);

	#endif
}

void calculateColorBleed(float2 sampleScreenSpacePosition, float sampleDepth, float3 pixelViewSpaceNormal, float3 sampleViewSpaceNormal, float3 sampleViewSpacePosition, float3 pixelViewSpacePosition, float farPlane, int i, out float3 gi, out float giCount) {

	gi = float3(0.0f, 0.0f, 0.0f);
	giCount = 0.0f;

#if SHADER_API_D3D9
	if ((giQuality == 1) ||
		((((i + 1) / giQuality) * giQuality == (i + 1)))) {
#else
	if (giQuality == 1 || 
		(giQuality == 2 && (((i + 1) & 1) == 0)) || 
		(giQuality == 4 && (((i + 1) & 3) == 0))) {
#endif
		float3 sampleViewSpacePosition = getViewSpacePositionUsingDepth(sampleScreenSpacePosition, sampleDepth / -farPlane);

		float3 cameraRay = sampleViewSpacePosition - pixelViewSpacePosition;
		float sampleDistance = length(cameraRay);
		cameraRay = normalize(cameraRay);

		float cosineNormals = dot(sampleViewSpaceNormal, pixelViewSpaceNormal);

		if (sampleDistance > radius || cosineNormals > 0.95f) {
			gi = float3(1, 1, 1);
		} else {

			float weight = pow(1.0f - (sampleDistance / radius), 2);

			if (giBackfaces == 0) {
				float _alpha = min(0.0f, -dot(sampleViewSpaceNormal, cameraRay));
				weight *= (_alpha + 1.0f) / 2.0f;
			}

			if (giSelfOcclusionFix == 2) {
				weight *= max(0.0f, dot(pixelViewSpaceNormal, cameraRay));
			}
			else if (giSelfOcclusionFix == 1) {
				if (dot(pixelViewSpaceNormal, cameraRay) < 0.0f) weight = 0.0f;
			}

			if (weight < 0.1f) {
				gi = float3(1, 1, 1);
			}
			else {

#if UNITY_UV_STARTS_AT_TOP
				if (_MainTex_TexelSize.y < 0)
					sampleScreenSpacePosition.y = 1.0f - sampleScreenSpacePosition.y;
#endif

#if UNITY_VERSION < 560
#ifdef WFORCE_VAO_OPENGL_OFF
				if (flipY != 0) {
					sampleScreenSpacePosition.y = 1.0f - sampleScreenSpacePosition.y;
				}
#endif
#endif

				float3 color = tex2Dlod(cbInputTex, float4(sampleScreenSpacePosition.xy, 0, 0)).rgb;

				gi = lerp(float3(1, 1, 1), color, weight);
			}
		}

		giCount = 1.0f;
	}
}
		
void getSampleDepthNormal(int depthEncodingType, float2 sampleScreenSpacePosition, float farPlane, out float sampleDepth, out float3 normal) {

	#ifdef WFORCE_VAO_COLORBLEED_OFF

		if (depthEncodingType == 1) {
			// Already in linear <0;1> range
			sampleDepth = tex2Dlod(depthNormalsTexture4, float4(sampleScreenSpacePosition.xy, 0, 0)).r;
		} else if (depthEncodingType == 2) {
			// Already in linear <0;1> range
			sampleDepth = tex2Dlod(depthNormalsTexture2, float4(sampleScreenSpacePosition.xy, 0, 0)).r;
		} else if (depthEncodingType == 3) {
			// Decode from RG (16 bits)
			sampleDepth = DecodeFloatRG(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.xy, 0, 0)).zw);
		} else {
			// Non linear depth buffer
			sampleDepth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(sampleScreenSpacePosition.xy, 0, 0)).r);
		}

		normal = float3(0, 0, -1);

	#else 
		if (depthEncodingType == 1) {
			// Already in linear <0;1> range
			DecodeDepthNormal(tex2Dlod(depthNormalsTexture4, float4(sampleScreenSpacePosition, 0.0f, 0.0f)), sampleDepth, normal);
		} else if (depthEncodingType == 2) {
			// Already in linear <0;1> range
			DecodeDepthNormal(tex2Dlod(depthNormalsTexture2, float4(sampleScreenSpacePosition, 0.0f, 0.0f)), sampleDepth, normal);
		} else if (depthEncodingType == 3) {
			// Decode from RG (16 bits)
			if (normalsSource == 1) {
				DecodeDepthNormal(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.xy, 0, 0)), sampleDepth, normal);
			} else {
				decodeDepth(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition, 0.0f, 0.0f)), sampleDepth);
				normal = calculateNormal(sampleScreenSpacePosition);
			}
		} else {
			// Non linear depth buffer
			sampleDepth = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(sampleScreenSpacePosition.xy, 0, 0)).r);
			if (normalsSource == 1) {
				normal = fetchNormal(sampleScreenSpacePosition);
			} else {
				normal = calculateNormal(sampleScreenSpacePosition);
			}

		}

	#endif

	// To view space depth
	sampleDepth = -sampleDepth * farPlane;

	float ux = max(sampleScreenSpacePosition.x - 1.0f, -sampleScreenSpacePosition.x);
	float uy = max(sampleScreenSpacePosition.y - 1.0f, -sampleScreenSpacePosition.y);
	float u = max(0.0f, max(ux, uy)) * 10.0f; //< Max suppresion achieved at 10% away from screen border
			
	if (u > 0.0f) {

		// Handle out of screen areas manually (we can't set clamp color for built-in depth buffers)
		sampleDepth -= lerp(0.0f, halfRadius * bordersIntensity * 10.0f, u);

#ifdef WFORCE_VAO_COLORBLEED_ON
		normal = float3(0, 0, -1);
#endif
	}

}

void getSampleDepthNormalDouble(int depthEncodingType, float4 sampleScreenSpacePosition, float farPlane, out float2 sampleDepth, out float3 normal, out float3 normal2) {

#ifdef WFORCE_VAO_COLORBLEED_OFF

	if (depthEncodingType == 1) {
		// Already in linear <0;1> range
		sampleDepth.x = tex2Dlod(depthNormalsTexture4, float4(sampleScreenSpacePosition.xy, 0, 0)).r;
		sampleDepth.y = tex2Dlod(depthNormalsTexture4, float4(sampleScreenSpacePosition.zw, 0, 0)).r;
	}
	else if (depthEncodingType == 2) {
		// Already in linear <0;1> range
		sampleDepth.x = tex2Dlod(depthNormalsTexture2, float4(sampleScreenSpacePosition.xy, 0, 0)).r;
		sampleDepth.y = tex2Dlod(depthNormalsTexture2, float4(sampleScreenSpacePosition.zw, 0, 0)).r;
	}
	else if (depthEncodingType == 3) {
		// Decode from RG (16 bits)
		sampleDepth.x = DecodeFloatRG(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.xy, 0, 0)).zw);
		sampleDepth.y = DecodeFloatRG(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.zw, 0, 0)).zw);
	}
	else {
		// Non linear depth buffer
		sampleDepth.x = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(sampleScreenSpacePosition.xy, 0, 0)).r);
		sampleDepth.y = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(sampleScreenSpacePosition.zw, 0, 0)).r);
	}

	normal = float3(0, 0, -1);
	normal2 = float3(0, 0, -1);

#else 
	if (depthEncodingType == 1) {
		// Already in linear <0;1> range
		DecodeDepthNormal(tex2Dlod(depthNormalsTexture4, float4(sampleScreenSpacePosition.xy, 0.0f, 0.0f)), sampleDepth.x, normal);
		DecodeDepthNormal(tex2Dlod(depthNormalsTexture4, float4(sampleScreenSpacePosition.zw, 0.0f, 0.0f)), sampleDepth.y, normal2);
	}
	else if (depthEncodingType == 2) {
		// Already in linear <0;1> range
		DecodeDepthNormal(tex2Dlod(depthNormalsTexture2, float4(sampleScreenSpacePosition.xy, 0.0f, 0.0f)), sampleDepth.x, normal);
		DecodeDepthNormal(tex2Dlod(depthNormalsTexture2, float4(sampleScreenSpacePosition.zw, 0.0f, 0.0f)), sampleDepth.y, normal2);
	}
	else if (depthEncodingType == 3) {
		// Decode from RG (16 bits)
		if (normalsSource == 1) {
			DecodeDepthNormal(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.xy, 0, 0)), sampleDepth.x, normal);
			DecodeDepthNormal(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.zw, 0, 0)), sampleDepth.y, normal2);
		}
		else {
			decodeDepth(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.xy, 0.0f, 0.0f)), sampleDepth.x);
			decodeDepth(tex2Dlod(_CameraDepthNormalsTexture, float4(sampleScreenSpacePosition.zw, 0.0f, 0.0f)), sampleDepth.y);
			normal = calculateNormal(sampleScreenSpacePosition.xy);
			normal2 = calculateNormal(sampleScreenSpacePosition.zw);
		}
	}
	else {
		// Non linear depth buffer
		sampleDepth.x = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(sampleScreenSpacePosition.xy, 0, 0)).r);
		sampleDepth.y = Linear01Depth(tex2Dlod(_CameraDepthTexture, float4(sampleScreenSpacePosition.zw, 0, 0)).r);
		if (normalsSource == 1) {
			normal = fetchNormal(sampleScreenSpacePosition.xy);
			normal2 = fetchNormal(sampleScreenSpacePosition.zw);
		}
		else {
			normal = calculateNormal(sampleScreenSpacePosition.xy);
			normal2 = calculateNormal(sampleScreenSpacePosition.zw);
		}

	}

#endif

	// To view space depth
	sampleDepth = -sampleDepth * farPlane;

	// Handle out of screen areas manually (we can't set clamp color for built-in depth buffers)
	// Only do check for one of two samples to speed up (they are probably very close to each other)
	float ux = max(sampleScreenSpacePosition.x - 1.0f, -sampleScreenSpacePosition.x);
	float uy = max(sampleScreenSpacePosition.y - 1.0f, -sampleScreenSpacePosition.y);
	float u = max(0.0f, max(ux, uy)) * 10.0f; //< Max suppresion achieved at 10% away from screen border

	if (u > 0.0f) {

		sampleDepth -= lerp(0.0f, halfRadius * bordersIntensity * 10.0f, u);

#ifdef WFORCE_VAO_COLORBLEED_ON
		normal = float3(0, 0, -1);
		normal2 = float3(0, 0, -1);
#endif
	}

}

int GetDepthTextureToUse(float pixelDepth) {

	if (hierarchicalBufferEnabled != 0) {
		if (pixelDepth < quarterResBufferMaxDistance) {
			// Quarter-res depth buffer
			return 1;
		} else if (pixelDepth < halfResBufferMaxDistance) {
			// Half-res depth buffer
			return 2;
		}
	}

	if (useGBuffer == 0 && useDedicatedDepthBuffer == 0) {
		// DepthNormals texture
		return 3;
	}

	// Depth Texture
	return 4;
}

// ========================================================================
// Vertex shaders 
// ========================================================================


v2fShed vertShed(appdata_img v)
{
	v2fShed o;
	
    if (noVertexTransform != 0)
    {
        o.pos = v.vertex;
    }
    else
    {
#ifdef UNITY_SINGLE_PASS_STEREO
	
	    if (isImageEffectMode != 0 && useSPSRFriendlyTransform != 0) {
		    o.pos = (v.vertex);
		    o.pos.xy *= 2.0f;
		    o.pos.xy -= 1.0f;

    #if defined(WFORCE_VAO_OPENGL_OFF)
		    v.texcoord.y = 1.0f - v.texcoord.y;
    #endif
	    }
	    else {
		    o.pos = UnityObjectToClipPos(v.vertex);
	    }
#else
        o.pos = UnityObjectToClipPos(v.vertex);
#endif
    }

	#ifdef UNITY_SINGLE_PASS_STEREO
		o.uv = UnityStereoTransformScreenSpaceTex(v.texcoord);
	#else
	if (noVertexTransform == 0) {
		o.uv = TRANSFORM_TEX(v.texcoord, _CameraDepthNormalsTexture);
	} else {
		o.uv = v.texcoord;
	}
	#endif

	#if UNITY_UV_STARTS_AT_TOP
	if (_MainTex_TexelSize.y < 0)
		o.uv.y = 1.0f - o.uv.y;
	#endif

    #if !defined(WFORCE_VAO_OPENGL_OFF)
    if (noVertexTransform != 0)
    {
        o.uv.y = 1.0f - o.uv.y;
    }
    #endif
				
	#ifdef UNITY_SINGLE_PASS_STEREO
		float2 tempUV1 = float2(o.uv.x * 2.0f, o.uv.y);
		float2 tempUV2 = float2(o.uv.x * 2.0f - 1.0f, o.uv.y);
		o.shed = mul(invProjMatrix, float4(tempUV1 * 2.0f - 1.0f, 1.0f, 1.0f));
		o.shed /= o.shed.w;
		o.shed2 = mul(invProjMatrix, float4(tempUV2 * 2.0f - 1.0f, 1.0f, 1.0f));
		o.shed2 /= o.shed2.w;
	#else
		o.shed = mul(invProjMatrix, float4(o.uv* 2.0f - 1.0f, 1.0f, 1.0f));
		o.shed /= o.shed.w;
	#endif

	return o;
}
	

v2fSingle vertSPSRCopy(appdata_img v)
{
v2fSingle o;
			
#if UNITY_VERSION < 201800
#ifdef UNITY_SINGLE_PASS_STEREO
		o.pos = (v.vertex);
		o.pos.xy *= 2.0f;
		o.pos.xy -= 1.0f;

#if defined(WFORCE_VAO_OPENGL_OFF)
		v.texcoord.y = 1.0f - v.texcoord.y;
#endif

#else
	o.pos = UnityObjectToClipPos(v.vertex);
#endif

	#ifdef UNITY_SINGLE_PASS_STEREO
	    o.uv = UnityStereoTransformScreenSpaceTex(v.texcoord);
	#else
		o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
	#endif

#else

    o.pos = UnityObjectToClipPos(v.vertex);
    o.uv = TRANSFORM_TEX(v.texcoord.xy, _MainTex);

#endif
    
	return o;
}
v2fSingle vertSingle(appdata_img v)
{
	v2fSingle o;
			
#ifdef UNITY_SINGLE_PASS_STEREO
	if (isImageEffectMode != 0 && useSPSRFriendlyTransform != 0) {
		o.pos = (v.vertex);
		o.pos.xy *= 2.0f;
		o.pos.xy -= 1.0f;

#if defined(WFORCE_VAO_OPENGL_OFF)
		v.texcoord.y = 1.0f - v.texcoord.y;
#endif
	}
	else {
		o.pos = UnityObjectToClipPos(v.vertex);
	}
#else
	o.pos = UnityObjectToClipPos(v.vertex);
#endif

	#ifdef UNITY_SINGLE_PASS_STEREO
		o.uv = UnityStereoTransformScreenSpaceTex(v.texcoord);
	#else
		o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
	#endif
	return o;
}

v2fDouble vertDouble(appdata_img v)
{
	v2fDouble o;
			
#ifdef UNITY_SINGLE_PASS_STEREO
	if (isImageEffectMode != 0 && useSPSRFriendlyTransform != 0) {
		o.pos = (v.vertex);
		o.pos.xy *= 2.0f;
		o.pos.xy -= 1.0f;

#if defined(WFORCE_VAO_OPENGL_OFF)
		v.texcoord.y = 1.0f - v.texcoord.y;
#endif
	}
	else {
		o.pos = UnityObjectToClipPos(v.vertex);
	}
#else
	o.pos = UnityObjectToClipPos(v.vertex);
#endif

	#ifdef UNITY_SINGLE_PASS_STEREO
	float2 temp = UnityStereoTransformScreenSpaceTex(v.texcoord);
	#else
	float2 temp = TRANSFORM_TEX(v.texcoord, _MainTex);
	#endif
	o.uv[0] = temp;
	o.uv[1] = temp;
				
	#if UNITY_UV_STARTS_AT_TOP
	if (_MainTex_TexelSize.y < 0)
		o.uv[1].y = 1.0f - o.uv[1].y;
	#endif

	#if UNITY_VERSION < 560
	#ifdef WFORCE_VAO_OPENGL_OFF
	if (flipY != 0) {
		o.uv[0].y = 1.0f - o.uv[1].y;
	}
	#endif
	#endif

	return o;
}

v2fDouble vertDoubleTexCopy(appdata_img v)
{
	v2fDouble o;
	o.pos = v.vertex;

	#ifdef UNITY_SINGLE_PASS_STEREO
	float2 temp = UnityStereoTransformScreenSpaceTex(v.texcoord);
	#else
	float2 temp = v.texcoord;
	#endif
			
	o.uv[0] = temp;
	o.uv[1] = temp;
				
	#if UNITY_UV_STARTS_AT_TOP
	if (_MainTex_TexelSize.y < 0)
		o.uv[1].y = 1.0f - o.uv[1].y;
	#endif

	#if !defined(WFORCE_VAO_OPENGL_OFF)
	o.uv[0].y = 1.0f - o.uv[0].y;
	o.uv[1].y = o.uv[0].y;
	#endif

	return o;
}

// ========================================================================
// Fragment Shaders 
// ========================================================================

void calculateRaycast(int kernelLength, int start, float AOhalfRadius, float AOhalfRadiusSquared, float3 tangentSphereCenterViewSpacePosition, float2 rotationAngleCosSin, float4x4 rotationMatrix, float farPlane, int depthTextureType, float3 pixelViewSpacePosition, float3 pixelViewSpaceNormal, out float kernelWeight, out float acc, out float3 gi, out float giCount, int enableColorbleed) {

	float2 sampleDepth = 0.0f;
	float3 sampleViewSpaceNormal = float3(0.0f, 0.0f, -1.0f);
	float3 sampleViewSpaceNormal2 = float3(0.0f, 0.0f, -1.0f);
    kernelWeight = (float) (kernelLength * 2);
	acc = 0.0f;
	gi = float3(0.0f, 0.0f, 0.0f);
	giCount = 0.0f;

#if defined(WFORCE_CAN_UNROLL) && !defined(WFORCE_VAO_COLORBLEED_ON) && !defined(SHADER_API_D3D9)	
	WFORCE_UNROLL(4)
#endif
		for (int i = 0; i < kernelLength; i++) {

#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)
			float4 rotatedSample = tex2Dlod(temporalSamples, float4((float2(i + start, 0.0f) * temporalTexelSizeRcp.xy) + temporalTexelSizeRcp.zw, 0.0f, 0.0f)) * AOhalfRadius;
#else
			float4 sam = samples[i + start] * AOhalfRadius;
			float4 rotatedSample = mul(rotationMatrix, sam);
#endif

			float3 sampleViewSpacePosition = tangentSphereCenterViewSpacePosition + float3(rotatedSample.xy, 0);
			float3 sampleViewSpacePosition2 = tangentSphereCenterViewSpacePosition + float3(rotatedSample.zw, 0);
			float2 sampleScreenSpacePosition = ((sampleViewSpacePosition.xy * screenProjection.xy +
				sampleViewSpacePosition.z * screenProjection.zw) / sampleViewSpacePosition.z) + 0.5f;

			float2 sampleScreenSpacePosition2 = ((sampleViewSpacePosition2.xy * screenProjection.xy +
				sampleViewSpacePosition2.z * screenProjection.zw) / sampleViewSpacePosition2.z) + 0.5f;

			sampleScreenSpacePosition = UnityStereoTransformScreenSpaceTex(sampleScreenSpacePosition);
			sampleScreenSpacePosition2 = UnityStereoTransformScreenSpaceTex(sampleScreenSpacePosition2);

			getSampleDepthNormalDouble(depthTextureType, float4(sampleScreenSpacePosition, sampleScreenSpacePosition2), farPlane, sampleDepth, sampleViewSpaceNormal, sampleViewSpaceNormal2);

			float3 samplePointViewSpacePosition = getViewSpacePositionUsingDepth(sampleScreenSpacePosition, sampleDepth.x / -farPlane);
			float3 samplePointViewSpacePosition2 = getViewSpacePositionUsingDepth(sampleScreenSpacePosition2, sampleDepth.y / -farPlane);

			float3 ray = samplePointViewSpacePosition - pixelViewSpacePosition;
			float3 ray2 = samplePointViewSpacePosition2 - pixelViewSpacePosition;
			acc += max(0.0f, 1.0f - (max(0.0f, dot(ray, pixelViewSpaceNormal) + ssaoBias * pixelViewSpacePosition.z)) / (dot(ray, ray) + 0.0001f));
			acc += max(0.0f, 1.0f - (max(0.0f, dot(ray2, pixelViewSpaceNormal) + ssaoBias * pixelViewSpacePosition.z)) / (dot(ray2, ray2) + 0.0001f));
					
			// Colorbleeding calculation
			#if defined(WFORCE_VAO_COLORBLEED_ON) && !defined(SHADER_API_D3D9)
					
				if (enableColorbleed != 0) {
					float3 sampleGi = float3(0.0f, 0.0f, 0.0f);
					float sampleGiCount = 0.0f;
					float3 sampleGi2 = float3(0.0f, 0.0f, 0.0f);
					float sampleGiCount2 = 0.0f;

					calculateColorBleed(sampleScreenSpacePosition, sampleDepth.x, pixelViewSpaceNormal, sampleViewSpaceNormal, sampleViewSpacePosition, pixelViewSpacePosition, farPlane, i, sampleGi, sampleGiCount);
					calculateColorBleed(sampleScreenSpacePosition2, sampleDepth.y, pixelViewSpaceNormal, sampleViewSpaceNormal2, sampleViewSpacePosition2, pixelViewSpacePosition, farPlane, i, sampleGi2, sampleGiCount2);

					gi += sampleGi;
					giCount += sampleGiCount;
					gi += sampleGi2;
					giCount += sampleGiCount2;
				}

			#endif

		}
			
}

static const int oneSppSamplesIndices[9] =
{
	0,  //0
	18,  //1 - 2
	54,  //2 - 4
	54, // 3 - 4
	126, // 4 - 8
	126, // 5 - 8
	126, // 6 - 8
	126, // 7 - 8
	270, // 8 - 16
};

void calculateVAO(int kernelLength, int start, float AOhalfRadius, float AOhalfRadiusSquared, float3 tangentSphereCenterViewSpacePosition, float2 rotationAngleCosSin, float4x4 rotationMatrix, float farPlane, int depthTextureType, float3 pixelViewSpacePosition, float3 pixelViewSpaceNormal, out float kernelWeight, out float acc, out float3 gi, out float giCount, int enableColorbleed) {

	float2 sampleDepth = 0.0f;
	float3 sampleViewSpaceNormal = float3(0.0f, 0.0f, -1.0f);
	float3 sampleViewSpaceNormal2 = float3(0.0f, 0.0f, -1.0f);
	kernelWeight = 0.0f;
	acc = 0.0f;
	gi = float3(0.0f, 0.0f, 0.0f);
	giCount = 0.0f;

	float dotTangentSphereCenterViewSpacePosition = dot(tangentSphereCenterViewSpacePosition, tangentSphereCenterViewSpacePosition);

#if defined(WFORCE_CAN_UNROLL) && !defined(WFORCE_VAO_COLORBLEED_ON)	
	WFORCE_UNROLL(4)
#endif
	for (int i = 0; i < kernelLength; i++) {

#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)
		float4 rotatedSample = tex2Dlod(temporalSamples, float4((float2(i + start, 0.0f) * temporalTexelSizeRcp.xy) + temporalTexelSizeRcp.zw, 0.0f, 0.0f)) * AOhalfRadius;
#else
		float4 sam = samples[i + start] * AOhalfRadius;
		float4 rotatedSample = mul(rotationMatrix, sam);
#endif

		float3 sampleViewSpacePosition = tangentSphereCenterViewSpacePosition + float3(rotatedSample.xy, 0);
		float3 sampleViewSpacePosition2 = tangentSphereCenterViewSpacePosition + float3(rotatedSample.zw, 0);
		float2 sampleScreenSpacePosition = ((sampleViewSpacePosition.xy * screenProjection.xy +
			sampleViewSpacePosition.z * screenProjection.zw) / sampleViewSpacePosition.z) + 0.5f;

		float2 sampleScreenSpacePosition2 = ((sampleViewSpacePosition2.xy * screenProjection.xy +
			sampleViewSpacePosition2.z * screenProjection.zw) / sampleViewSpacePosition2.z) + 0.5f;

		sampleScreenSpacePosition = UnityStereoTransformScreenSpaceTex(sampleScreenSpacePosition);
		sampleScreenSpacePosition2 = UnityStereoTransformScreenSpaceTex(sampleScreenSpacePosition2);

		getSampleDepthNormalDouble(depthTextureType, float4(sampleScreenSpacePosition, sampleScreenSpacePosition2), farPlane, sampleDepth, sampleViewSpaceNormal, sampleViewSpaceNormal2);

		float3x3 ray = float3x3(normalize(sampleViewSpacePosition),
								normalize(sampleViewSpacePosition2),
								0.0f, 0.0f, 0.0f);

		float2 tca = mul(ray, tangentSphereCenterViewSpacePosition);
		float2 d2 = dotTangentSphereCenterViewSpacePosition - (tca * tca);
		float2 diff = AOhalfRadiusSquared - d2;
				
		float2 thc = vaoSqrt(diff);
		float2 entryDepth = tca - thc;
		float2 exitDepth = tca + thc;

		entryDepth = float2(entryDepth.x * ray[0][2], entryDepth.y * ray[1][2]);
		exitDepth = float2(exitDepth.x * ray[0][2], exitDepth.y * ray[1][2]);

		float2 pipeLength = entryDepth - exitDepth;

		if (diff.x > 0.001f) {
			kernelWeight += pipeLength.x;

			if (sampleDepth.x > entryDepth.x) {
				acc += pipeLength.x * max(0.0f, (1.0f - (aoThickness * AOhalfRadius / (max(aoThickness, (sampleDepth.x - entryDepth.x))))));
			}
			else if (sampleDepth.x < exitDepth.x) {
				acc += pipeLength.x;
			}
			else {
				acc += entryDepth.x - sampleDepth.x;
			}
		}

		if (diff.y > 0.001f) {
			kernelWeight += pipeLength.y;

			if (sampleDepth.y > entryDepth.y) {
				acc += pipeLength.y * max(0.0f, (1.0f - (aoThickness * AOhalfRadius / (max(aoThickness, (sampleDepth.y - entryDepth.y))))));
			}
			else if (sampleDepth.y < exitDepth.y) {
				acc += pipeLength.y;
			}
			else {
				acc += entryDepth.y - sampleDepth.y;
			}
		}

		// Colorbleeding calculation
#if defined(WFORCE_VAO_COLORBLEED_ON) && !defined(SHADER_API_D3D9)

		if (enableColorbleed != 0) {
			float3 sampleGi = float3(0.0f, 0.0f, 0.0f);
			float sampleGiCount = 0.0f;
			float3 sampleGi2 = float3(0.0f, 0.0f, 0.0f);
			float sampleGiCount2 = 0.0f;

			calculateColorBleed(sampleScreenSpacePosition, sampleDepth.x, pixelViewSpaceNormal, sampleViewSpaceNormal, sampleViewSpacePosition, pixelViewSpacePosition, farPlane, i, sampleGi, sampleGiCount);
			calculateColorBleed(sampleScreenSpacePosition2, sampleDepth.y, pixelViewSpaceNormal, sampleViewSpaceNormal2, sampleViewSpacePosition2, pixelViewSpacePosition, farPlane, i, sampleGi2, sampleGiCount2);

			gi += sampleGi;
			giCount += sampleGiCount;
			gi += sampleGi2;
			giCount += sampleGiCount2;
		}
#endif

	}

}

void calculateAO(int algorithm, int kernelLength, int start, float AOhalfRadius, float AOhalfRadiusSquared, float3 tangentSphereCenterViewSpacePosition, float2 rotationAngleCosSin, float4x4 rotationMatrix, float farPlane, int depthTextureType, float3 pixelViewSpacePosition, float3 pixelViewSpaceNormal, out float kernelWeight, out float acc, out float3 gi, out float giCount, int enableColorbleed) {

	float _kernelWeight = 0.0f; float _acc = 0.0f; float3 _gi = float3(0.0f, 0.0f, 0.0f); float _giCount = 0.0f;

	if (algorithm == WFORCE_STANDARD_VAO) {
		calculateVAO(kernelLength, start, AOhalfRadius, AOhalfRadiusSquared, tangentSphereCenterViewSpacePosition, rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, _kernelWeight, _acc, _gi, _giCount, enableColorbleed);
	} else {
        #if !defined(SHADER_API_D3D9)
        calculateRaycast(kernelLength, start, AOhalfRadius, AOhalfRadiusSquared, tangentSphereCenterViewSpacePosition, rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, _kernelWeight, _acc, _gi, _giCount, enableColorbleed);
        #endif
    }

	kernelWeight = _kernelWeight; acc = _acc; gi = _gi; giCount = _giCount;
}

void getAdaptiveSamplesCount(float pixelViewSpacePositionz, out int samplesCount, out int samplesStart)
{
    float u = saturate((pixelViewSpacePositionz - adaptiveMin) / (adaptiveMax - adaptiveMin));
        
    if (u < 0.25f)
    {
        if (u < 0.125f)
        {
            samplesCount = 32;
            samplesStart = 32;
        }
        else
        {
            samplesCount = 16;
            samplesStart = 48;
        }
    }
    else
    {
        if (u < 0.5f)
        {
            samplesCount = 8;
            samplesStart = 56;
        }
        else
        {
            samplesCount = 4;
            samplesStart = 60;
        }
    }

    samplesCount = min(samplesCount, maxSamplesCount);
    samplesStart = max(samplesCount, samplesStartIndex);

}

WFORCE_VAO_MAIN_PASS_RETURN_TYPE ao(v2fShed i, int algorithm, int doQuarterRadius, int quarterRadiusSamplesCount, int doHalfRadius, int halfRadiusSamplesCount, int isCullingPrepass, int cullingPrepassType, int adaptive, int kernelLength) {
	float pixelDepth = 0.0f;
	float3 pixelViewSpaceNormal = float3(0.0f, 0.0f, 0.0f);
	float kernelWeight = 0.0f;
    int start = samplesStartIndex;
    int sampNum = kernelLength;
	float distanceFalloffFactor = 0.0f;
	float acc = 0.0f;
    float aoEstimation = 0.0f;

#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)
	WFORCE_VAO_MAIN_PASS_RETURN_TYPE whiteResult;
	WFORCE_VAO_MAIN_PASS_RETURN_TYPE blackResult;

	whiteResult.aoDepth = half4(1.0f, 1.0f, 1.0f, 1.0f);
	whiteResult.history = half4(1.0f, 1.0f, 1.0f, 1.0f);

	blackResult.aoDepth = half4(0.0f, 1.0f, 1.0f, 1.0f);
	blackResult.history = half4(0.0f, 0.0f, 0.0f, 0.0f);

    #ifdef WFORCE_VAO_COLORBLEED_ON
	    blackResult.gi = half4(0.0f, 0.0f, 0.0f, 0.0f);
	    blackResult.gi2 = half4(0.0f, 0.0f, 0.0f, 0.0f);
	    blackResult.gi3 = half4(0.0f, 0.0f, 0.0f, 0.0f);

        whiteResult.gi = half4(0.0f, 0.0f, 0.0f, 0.0f);
	    whiteResult.gi2 = half4(0.0f, 0.0f, 0.0f, 0.0f);
	    whiteResult.gi3 = half4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif
#else
	WFORCE_VAO_MAIN_PASS_RETURN_TYPE whiteResult = WFORCE_VAO_WHITE;
	WFORCE_VAO_MAIN_PASS_RETURN_TYPE blackResult = WFORCE_VAO_BLACK;
#endif

	float3 gi = float3(0.0f, 0.0f, 0.0f);
	float giCount = 0.0f;
			
	// Greedy culling pre-pass check
	if (isCullingPrepass == 0) {
		if (cullingPrepassType != 0) {
			aoEstimation = tex2Dlod(cullingPrepassTexture, float4(i.uv, 0, 0)).r;

			if (cullingPrepassType == 1 && aoEstimation >= 1.0f) return whiteResult;
		}
	}
			
	if (normalsSource == 1) {
		fetchDepthNormal(i.uv, pixelDepth, pixelViewSpaceNormal);
	} else {
		fetchDepthCalculateNormal(i.uv, pixelDepth, pixelViewSpaceNormal);
	}

	if (pixelDepth > subpixelRadiusCutoffDepth) return whiteResult; 
			
	float3 pixelViewSpacePosition = (i.shed.rgb * pixelDepth);
	#ifdef UNITY_SINGLE_PASS_STEREO
	if (i.uv.x > .5f) {
		pixelViewSpacePosition = (i.shed2.rgb * pixelDepth);
	}
	#endif
				
	// Careful culling pre-pass check and adaptive sampling
	if (isCullingPrepass == 0) {

		if (cullingPrepassType == 2 && aoEstimation >= 1.0f) {
			// Careful culling prepass
            start = fourSamplesStartIndex;
            sampNum = 4;
        } else {
			// Adaptive sampling
            if (adaptive != 0) getAdaptiveSamplesCount(pixelViewSpacePosition.z, sampNum, start);
        }
	} else {
        start = fourSamplesStartIndex;
    }

	// Distance falloff
	if (minRadiusEnabled && pixelViewSpacePosition.z < minRadiusCutoffDepth) {
		distanceFalloffFactor = smoothstep(minRadiusCutoffDepth, minRadiusCutoffDepth - minRadiusSoftness, pixelViewSpacePosition.z);
		if (distanceFalloffFactor == 1.0f) return whiteResult;
	}

#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)

		float2 previousPos;
		float4 previousPixelScreenSpacePosition;

		if (enableReprojection == 0) {
			previousPos = i.uv;
		} else {
			if (useUnityMotionVectors != 0) {
			    previousPos = i.uv - tex2Dlod(_CameraMotionVectorsTexture, float4(i.uv, 0, 0)).rg;
            } else {

                previousPixelScreenSpacePosition = mul(motionVectorsMatrix, float4(pixelViewSpacePosition, 1.0f));
                previousPixelScreenSpacePosition.xyz /= previousPixelScreenSpacePosition.w;
				previousPos = (previousPixelScreenSpacePosition.xy) * 0.5 + 0.5f;

#ifdef UNITY_SINGLE_PASS_STEREO
				if (i.uv.x > 0.5f) {
					previousPos.x = previousPos.x * 0.5f + 0.5f;
				}
				else {
					previousPos.x = previousPos.x * 0.5f;
				}
#endif
			}
		}
		
		if (previousPos.x > 1.0f || previousPos.x < 0.0f)
			previousPos = float2(-1.0f, -1.0f);

		if (previousPos.y > 1.0f || previousPos.y < 0.0f)
			previousPos = float2(-1.0f, -1.0f);

		float4 aoHistory = tex2Dlod(aoHistoryTexture, float4(previousPos, 0, 0)).rgba;

#ifdef WFORCE_VAO_COLORBLEED_ON
		float3 giPrevious = tex2Dlod(giHistoryTexture, float4(previousPos, 0, 0)).rgb;
		float3 giPrevious2 = tex2Dlod(gi2HistoryTexture, float4(previousPos, 0, 0)).rgb;
		float3 giPrevious3 = tex2Dlod(gi3HistoryTexture, float4(previousPos, 0, 0)).rgb;
#endif

		if (abs(1.0f - (pixelDepth / aoHistory.a)) > 0.05f)
			previousPos = float2(-1.0f, -1.0f);

		int rejectReprojection = (previousPos.x == -1.0f && previousPos.y == -1.0f) ? 1 : 0;

		sampNum = max(sampNum / 4, 2);

		int noiseIdx = dot(int2(fmod(i.uv.xy * inputTexDimensions.xy, 3)), int2(1, 3));
		int samplesCountToUse = sampNum / 2;
		start = oneSppSamplesIndices[min(8, samplesCountToUse)] + (noiseIdx * (samplesCountToUse)) + int(frameNumber * (samplesCountToUse * 9));
#endif

	// Max radius limit
	if (maxRadiusEnabled && pixelViewSpacePosition.z > maxRadiusCutoffDepth) {
		radius = getRadiusForDepthAndScreenRadius(pixelViewSpacePosition.z, maxRadiusOnScreen);
		halfRadius = 0.5f * radius;
		halfRadiusSquared = halfRadius * halfRadius;
	}

	// Hierarchical buffer level
	int depthTextureType = GetDepthTextureToUse(pixelDepth);

	float3 tangentSphereCenterViewSpacePosition = pixelViewSpacePosition + (pixelViewSpaceNormal * halfRadius);

	float farPlane = getFarPlane(UseCameraFarPlane);

	float2 rotationAngleCosSin = tex2Dlod(noiseTexture, float4(i.uv * noiseTexelSizeRcp, 0.0f, 0.0f));
	float4x4 rotationMatrix = float4x4(rotationAngleCosSin.x, rotationAngleCosSin.y, 0, 0,
		-rotationAngleCosSin.y, rotationAngleCosSin.x, 0, 0,
		0, 0, rotationAngleCosSin.x, rotationAngleCosSin.y,
		0, 0, -rotationAngleCosSin.y, rotationAngleCosSin.x);

	// Quarter radius detail
	float eightRadius = halfRadius * 0.25f;
	float quarterResAcc;
	float quarterResWeight;
#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)
	if (doQuarterRadius != 0 && isCullingPrepass == 0) calculateAO(algorithm, 1, 18 + (noiseIdx) + (frameNumber * 9), eightRadius, eightRadius * eightRadius, pixelViewSpacePosition + (pixelViewSpaceNormal * eightRadius), rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, quarterResWeight, quarterResAcc, gi, giCount, 0);
#else
	if (doQuarterRadius != 0 && isCullingPrepass == 0) calculateAO(algorithm, quarterRadiusSamplesCount / 2, quarterRadiusSamplesOffset, eightRadius, eightRadius * eightRadius, pixelViewSpacePosition + (pixelViewSpaceNormal * eightRadius), rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, quarterResWeight, quarterResAcc, gi, giCount, 0);
#endif

	// Half radius detail
	float quarterRadius = halfRadius * 0.5f;
	float halfResAcc;
	float halfResWeight;
#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)
	if (doHalfRadius != 0 && isCullingPrepass == 0) calculateAO(algorithm, 1, 18 + (noiseIdx)+(frameNumber * 9), quarterRadius, quarterRadius * quarterRadius, pixelViewSpacePosition + (pixelViewSpaceNormal * quarterRadius), rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, halfResWeight, halfResAcc, gi, giCount, 0);
#else
    if (doHalfRadius != 0 && isCullingPrepass == 0) calculateAO(algorithm, halfRadiusSamplesCount / 2, halfRadiusSamplesOffset, quarterRadius, quarterRadius * quarterRadius, pixelViewSpacePosition + (pixelViewSpaceNormal * quarterRadius), rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, halfResWeight, halfResAcc, gi, giCount, 0);
#endif

    if (sampNum == 2)
        calculateAO(algorithm, 1, start, halfRadius, halfRadiusSquared, tangentSphereCenterViewSpacePosition, rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, kernelWeight, acc, gi, giCount, 1);
    else if (sampNum == 4)
        calculateAO(algorithm, 2, start, halfRadius, halfRadiusSquared, tangentSphereCenterViewSpacePosition, rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, kernelWeight, acc, gi, giCount, 1);
    else
    {
        int cycles = sampNum / 8;

        for (int cycle = 0; cycle < cycles; cycle++)
        {
            float _kernelWeight = 0.0f; float _acc = 0.0f; float3 _gi = float3(0.0f, 0.0f, 0.0f); float _giCount = 0.0f;

            calculateAO(algorithm, 4, start + (cycle * 4), halfRadius, halfRadiusSquared, tangentSphereCenterViewSpacePosition, rotationAngleCosSin, rotationMatrix, farPlane, depthTextureType, pixelViewSpacePosition, pixelViewSpaceNormal, _kernelWeight, _acc, _gi, _giCount, 1);

            kernelWeight += _kernelWeight; acc += _acc; gi += _gi; giCount += _giCount;
        }
    }

	// Finish early in culling pre-pass
#ifndef ENABLE_TEMPORAL_ACCUMULATION
	if (isCullingPrepass != 0) {

		#ifdef WFORCE_VAO_COLORBLEED_OFF

			if (acc == kernelWeight) 
				return whiteResult;
			else
				return blackResult;
		#else
					
			float prepassResult = 1.0f;
					
			if (acc != kernelWeight) 
				prepassResult = 0.0f;

			if (giCount > 0.0f) gi = gi / giCount; 

			if (gi.r < 1.0f || gi.g < 1.0f || gi.b < 1.0f) 
				prepassResult = 0.0f;

			return half4(prepassResult, 1.0f, 1.0f, 1.0f);
		#endif
	}
#endif

	if (kernelWeight > 0.0f){
		acc = acc / kernelWeight;
	} else {
		acc = 1.0f;
	}

	if (doHalfRadius != 0 && isCullingPrepass == 0 && halfResWeight > 0.0f) {
		acc *= lerp(1.0f, halfResAcc / halfResWeight, halfRadiusWeight);
	}

	if (doQuarterRadius != 0 && isCullingPrepass == 0 && quarterResWeight > 0.0f) {
		acc *= lerp(1.0f, quarterResAcc / quarterResWeight, quarterRadiusWeight);
	}

	float accFullPresence = 1.0f - vaoSqrt(1.0f-(acc*acc));
	if (acc > 0.999f) accFullPresence = 1.0f;

	// Presence
	acc = lerp(acc, accFullPresence, aoPresence);

	// Power
	acc = pow(acc, aoPower);

	// Distance falloff			
	acc = lerp(acc, 1.0f, distanceFalloffFactor);

	#ifdef WFORCE_VAO_COLORBLEED_ON
		if (giCount > 0.0f) gi = gi / giCount; 
		gi = pow(gi, giPower); 
	#endif

	// Luma sensitivity
	#ifdef WFORCE_VAO_COLORBLEED_OFF

		if (isLumaSensitive != 0) {
			if (hwBlendingEnabled != 0) {
				float3 mainColor = tex2Dlod(_MainTex, float4(i.uv, 0, 0)).rgb;
						
				if (useLogBufferInput != 0)
					mainColor = -log2(mainColor);

				float accIn = acc;
				applyLuma(mainColor, accIn, acc);
			}
		}
				
	#endif

#if defined(ENABLE_TEMPORAL_ACCUMULATION) && !defined(SHADER_API_D3D9)

			//acc = visualizeSamples(i.uv, start, sampNum).r;
			whiteResult.history = aoHistory;
			
			float4 tempHistory = float4(acc, whiteResult.history.r, whiteResult.history.g, whiteResult.history.b);
			float4 newHistory = ((rejectReprojection != 0 || historyReady == 0) ? float4(acc, acc, acc, acc) : tempHistory);

#ifdef WFORCE_VAO_COLORBLEED_OFF
			whiteResult.aoDepth = half4(dot(newHistory, float4(0.25f, 0.25f, 0.25f, 0.25f)), pixelDepth, 1, 1);
#else
			whiteResult.aoDepth = half4((gi + giPrevious + giPrevious2 + giPrevious3) * 0.25f, dot(newHistory, float4(0.25f, 0.25f, 0.25f, 0.25f)));
			whiteResult.gi = half4(gi, 1.0f);
			whiteResult.gi2 = half4(giPrevious, 1.0f);
			whiteResult.gi3 = half4(giPrevious2, 1.0f);
#endif
			whiteResult.history = float4(newHistory.rgb, pixelDepth);
			
			return whiteResult;
#else
#ifdef WFORCE_VAO_COLORBLEED_OFF
		return half4(acc, pixelDepth, 1, 1);
#else
		return half4(gi, acc);
#endif
#endif
	
}

half4 mixing(float4 color, half4 giao)
{	

	#ifdef WFORCE_VAO_COLORBLEED_ON
		half3 gi = giao.rgb;
	#endif

	float ao = giao.a;
			
	// Luma sensitivity
	if (isLumaSensitive != 0) {
		if (hwBlendingEnabled == 0) {
			float aoIn = ao;

			#ifdef WFORCE_VAO_COLORBLEED_ON
				float3 giIn = gi;
				applyLuma(color, aoIn, ao, giIn, gi);
			#else
				applyLuma(color.rgb, aoIn, ao);
			#endif
		}
	}

	#ifdef WFORCE_VAO_COLORBLEED_ON
			
		if (hwBlendingEnabled == 0) {
			gi = processGi(color, gi, ao);
		}

		if (outputAOOnly != 0) {
			return half4(gi, 1.0f);
		}

		color.rgb *= gi;
	#else

		if (outputAOOnly != 0) {
			color = half4(1.0f, 1.0f, 1.0f, 1.0f);
		}

		color.rgb *= ao+colorTint.rgb*(1.0f - ao);
	#endif

	if (useLogEmissiveBuffer != 0) {			
		// Encode for logarithmic emission buffer in LDR
		color.rgb = exp2(-color.rgb);
	}

	return color;
}
		
half4 mixingNoBlur(v2fDouble i)
{	
	float4 mainColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
			
	if (hwBlendingEnabled == 0 && useLogEmissiveBuffer == 0) {
		mainColor = tex2Dlod(_MainTex, float4(i.uv[0], 0.0f, 0.0f));
	}

	if (useLogEmissiveBuffer != 0) {			
		mainColor = tex2Dlod(emissionTexture, float4(i.uv[0], 0, 0)).rgba;
		mainColor.rgb = -log2(mainColor.rgb);
	}

	#ifdef WFORCE_VAO_COLORBLEED_ON
		return mixing(mainColor, tex2Dlod(textureAO, float4(i.uv[1], 0.0f, 0.0f)));
	#else
		return mixing(mainColor, half4(1.0f, 1.0f, 1.0f, tex2Dlod(textureAO, float4(i.uv[1], 0.0f, 0.0f)).r));
	#endif
}

half4 uniformBlur(v2fDouble input, const int increment)
{

	#ifdef WFORCE_VAO_COLORBLEED_ON
		float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);

		float weightGi = 4.0f;

		if (useFastBlur == 0) {
			weightGi = 9.0f;
		}

		float3 centerGi = tex2Dlod(textureAO, float4(input.uv[1], 0.0f, 0.0f)).rgb;
		float centerGiLuma = dot(float3(0.3f, 0.3f, 0.3f), centerGi);
	#else
		float acc = 0.0f;
	#endif

	float weight = 4.0f;

	if (useFastBlur == 0) {
		weight = 9.0f;
	}

	float farPlane = getFarPlane(UseCameraFarPlane);
			
	#ifdef WFORCE_VAO_COLORBLEED_OFF
		float centerDepth = -tex2Dlod(textureAO, float4(input.uv[1], 0, 0)).g * farPlane;
	#else
		float centerDepth = fetchDepth(input.uv[1], farPlane);
	#endif

	WFORCE_UNROLL(3)
	for (int i = -1; i <= 1; i+=increment) {
		WFORCE_UNROLL(3)
		for (int j = -1; j <= 1; j+=increment) {

			float2 offset = input.uv[1] + float2(float(i), float(j)) * texelSize;
					
			#ifdef WFORCE_VAO_COLORBLEED_ON

				float tapDepth = fetchDepth(offset, farPlane);

				half4 tapGiAO = tex2Dlod(textureAO, float4(offset, 0, 0));
				float tapLuma = dot(float3(0.3f, 0.3f, 0.3f), tapGiAO.rgb);
                        
				if (abs(tapDepth - centerDepth) > blurDepthThreshold) {
					weight -= 1.0f;
				} else {
					acc.a += tapGiAO.a;
				}

				if (giBlur != 3 && abs(centerGiLuma - tapLuma) > 0.2f) {
					weightGi -= 1.0f;
				} else {
					acc.rgb += tapGiAO.rgb;
				}

			#else
						
				float2 tap = tex2Dlod(textureAO, float4(offset, 0, 0));
				float tapDepth = -tap.g * farPlane;
					
				if (abs(tapDepth - centerDepth) > blurDepthThreshold) {
					weight -= 1.0f;
					continue;
				}

				acc += tap.r;

			#endif

		}
	}

	float result = 1.0f;

	half4 mainColor = half4(1.0f, 1.0f, 1.0f, 1.0f);
	if (hwBlendingEnabled == 0 && useLogEmissiveBuffer == 0) {
		mainColor = tex2Dlod(_MainTex, float4(input.uv[0], 0.0f, 0.0f));
	}
			
	if (useLogEmissiveBuffer != 0) {			
		mainColor = tex2Dlod(emissionTexture, float4(input.uv[0], 0, 0)).rgba;
		mainColor.rgb = -log2(mainColor.rgb);
	}

	#ifdef WFORCE_VAO_COLORBLEED_ON
		if (weight > 0.0f) result = acc.a / weight;
		float3 resultGi = float3(1.0f, 1.0f, 1.0f);
		if (weightGi > 0.0f) resultGi = acc.rgb / weightGi;
		return mixing(mainColor, half4(resultGi, result));
	#else
		if (weight > 0.0f) result = acc / weight;
		return mixing(mainColor, half4(1.0f, 1.0f, 1.0f, result));
	#endif
}

half4 enhancedBlur(v2fDouble input, int passIndex, const int blurSize)
{
	int idx = 0;
	float2 offset;
	float weight = gaussWeight;
	float farPlane = getFarPlane(UseCameraFarPlane);

	#ifdef WFORCE_VAO_COLORBLEED_OFF
		float acc = 0.0f;
		float centerDepth01 = tex2Dlod(textureAO, float4(input.uv[1], 0, 0)).g;
		float centerDepth = -centerDepth01 * farPlane;
	#else
		float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
		float centerDepth = fetchDepth(input.uv[1], farPlane);
		float weightGi = enhancedBlurSize * 2.0f + 1.0f;
		float centerGiLuma = dot(float3(0.3f, 0.3f, 0.3f), tex2Dlod(textureAO, float4(input.uv[1], 0, 0)).rgb);
	#endif

	WFORCE_UNROLL(17)
	for (int i = -blurSize; i <= blurSize; ++i) {
				
		if (passIndex == 1) 
			offset = input.uv[1] + float2(float(i) * texelSize.x, 0.0f);
		else 
			offset = input.uv[1] + float2(0.0f, float(i) * texelSize.y);
			
		#ifdef WFORCE_VAO_COLORBLEED_OFF
			float2 tapSample = tex2Dlod(textureAO, float4(offset, 0, 0));
			float tapDepth = -tapSample.g * farPlane;
			float tapAO = tapSample.r;
		#else
			float4 tapSample = tex2Dlod(textureAO, float4(offset, 0, 0));
			float tapDepth = fetchDepth(offset, farPlane);
			float tapAO = tapSample.a;
		#endif

        if (abs(tapDepth - centerDepth) < blurDepthThreshold) {
			acc += tapAO * gauss[idx].x;
		} else {
			weight -= gauss[idx].x;
		}

		#ifdef WFORCE_VAO_COLORBLEED_ON
			float tapLuma = dot(float3(0.3f, 0.3f, 0.3f), tapSample.rgb);

			if (giBlur != 3 && abs(tapLuma - centerGiLuma) > 0.2f) {
				weightGi -= 1.0f;
			} else {
				acc.rgb += tapSample.rgb;
			}
		#endif

		idx++;
	}

	float result = 1.0f;

	#ifdef WFORCE_VAO_COLORBLEED_OFF
		if (weight > 0.0f) result = acc / weight;
	#else
		float3 resultGi = float3(1.0f, 1.0f, 1.0f);
		if (weightGi > 0.0f) resultGi = acc.rgb / weightGi;
		if (weight > 0.0f) result = acc.a / weight;
	#endif
			
	if (passIndex == 2) {

		float4 mainColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
			
		if (hwBlendingEnabled == 0 && useLogEmissiveBuffer == 0) {
			mainColor = tex2Dlod(_MainTex, float4(input.uv[0], 0.0f, 0.0f));
		}

		if (useLogEmissiveBuffer != 0) {			
			mainColor = tex2Dlod(emissionTexture, float4(input.uv[0], 0, 0)).rgba;
			mainColor.rgb = -log2(mainColor.rgb);
		}

		#ifdef WFORCE_VAO_COLORBLEED_OFF
			return mixing(mainColor, half4(1.0f, 1.0f, 1.0f, result));
		#else
			return mixing(mainColor, half4(resultGi, result));
		#endif
	} else {
		#ifdef WFORCE_VAO_COLORBLEED_OFF
			return half4(result, centerDepth01, 0.0f, 0.0f);
		#else
			return half4(resultGi, result);
		#endif
	}
			
}

half4 selectEnhancedBlur(v2fDouble input, int passIndex)
{
	if (enhancedBlurSize == 1)
		return enhancedBlur(input, passIndex, 1);

	if (enhancedBlurSize == 2)
		return enhancedBlur(input, passIndex, 2);

	if (enhancedBlurSize == 3)
		return enhancedBlur(input, passIndex, 3);

	if (enhancedBlurSize == 4)
		return enhancedBlur(input, passIndex, 4);

	if (enhancedBlurSize == 5)
		return enhancedBlur(input, passIndex, 5);

	if (enhancedBlurSize == 6)
		return enhancedBlur(input, passIndex, 6);

	if (enhancedBlurSize == 7)
		return enhancedBlur(input, passIndex, 7);

	return enhancedBlur(input, passIndex, 8);
}

half4 blendLogGbuffer3(v2fDouble i) 
{
	float3 mainColor = tex2Dlod(emissionTexture, float4(i.uv[0], 0, 0)).rgba;
	mainColor = -log2(mainColor);
	mainColor *= tex2Dlod(occlusionTexture, float4(i.uv[0], 0.0f, 0.0f)).r;
	return half4(exp2(-mainColor.rgb), 1.0f);
}

AoOutput blendBeforeReflections(v2fDouble i) 
{
	AoOutput output;

	output.albedoAO = half4(0.0f, 0.0f, 0.0f, tex2Dlod(occlusionTexture, float4(i.uv[0], 0.0f, 0.0f)).r);
	output.emissionLighting = tex2Dlod(occlusionTexture, float4(i.uv[0], 0.0f, 0.0f)).rrrr;
			
	return output;
}

AoOutput blendBeforeReflectionsLog(v2fDouble i) 
{
	AoOutput output;

	float occlusion = tex2Dlod(occlusionTexture, float4(i.uv[0], 0.0f, 0.0f)).r;

	output.albedoAO = half4(0.0f, 0.0f, 0.0f, occlusion);
			
	float3 mainColor = tex2Dlod(emissionTexture, float4(i.uv[0], 0, 0)).rgba;
	mainColor = -log2(mainColor);
	mainColor *= occlusion;
	output.emissionLighting = half4(exp2(-mainColor.rgb), 1.0f);
			
	return output;
}

half4 blendAfterLightingLog(v2fDouble i) 
{
	float occlusion = tex2Dlod(occlusionTexture, float4(i.uv[0], 0.0f, 0.0f)).r;
	float3 mainColor = tex2Dlod(emissionTexture, float4(i.uv[0], 0, 0)).rgba;

	mainColor = -log2(mainColor);
	mainColor *= occlusion;

	return half4(exp2(-mainColor.rgb), 1.0f);
}
