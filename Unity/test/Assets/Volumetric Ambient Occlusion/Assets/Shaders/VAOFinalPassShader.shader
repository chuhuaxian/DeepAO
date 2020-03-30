// Copyright (c) 2016-2018 Jakub Boksansky - All Rights Reserved
// Volumetric Ambient Occlusion Unity Plugin 2.0

Shader "Hidden/Wilberforce/VAOFinalPassShader"
{
	Properties
	{
		_MainTex("Texture", 2D) = "white" {}
	}
		CGINCLUDE

		#pragma target 3.0

		#pragma multi_compile WFORCE_VAO_COLORBLEED_OFF WFORCE_VAO_COLORBLEED_ON	

		#include "VAO.cginc"

	ENDCG
		SubShader
		{

			Cull Off ZWrite Off ZTest Always

			// 0 - StandardBlurUniform
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble input) : SV_Target{ return uniformBlur(input, 1); }
				ENDCG }

			// 1 - StandardBlurUniformMultiplyBlend
			Pass{ Blend DstColor Zero // Multiplicative
				CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble input) : SV_Target{ return uniformBlur(input, 1); }
				ENDCG }

			// 2 - StandardBlurUniformFast
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble input) : SV_Target{ return uniformBlur(input, 2); }
				ENDCG }

			// 3 - StandardBlurUniformFastMultiplyBlend
			Pass{ Blend DstColor Zero // Multiplicative
				CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble input) : SV_Target{ return uniformBlur(input, 2); }
				ENDCG }

			// 4 - EnhancedBlurFirstPass
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return selectEnhancedBlur(i, 1); }
				ENDCG }

			// 5 - EnhancedBlurSecondPass
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return selectEnhancedBlur(i, 2); }
				ENDCG }

			// 6 - EnhancedBlurSecondPassMultiplyBlend
			Pass{ Blend DstColor Zero // Multiplicative
				CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return selectEnhancedBlur(i, 2); }
				ENDCG }

			// 7 - Mixing
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return mixingNoBlur(i); }
				ENDCG }

			// 8 - MixingMultiplyBlend
			Pass{ Blend DstColor Zero // Multiplicative
				CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return mixingNoBlur(i); }
				ENDCG }

			// 9 - DownscaleDepthNormalsPass
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return downscaleDepthNormals(i); }
				ENDCG }

			// 10 - Copy
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return tex2Dlod(_MainTex, float4(i.uv[1],0,0)); }
				ENDCG }

			// 11 - BlendAfterLightingLog
			Pass{ CGPROGRAM
				#pragma vertex vertDoubleTexCopy #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return blendAfterLightingLog(i); }
				ENDCG }

			// 12 - TexCopyImageEffectSPSR
			Pass{ CGPROGRAM
				#pragma vertex vertDouble #pragma fragment frag
				half4 frag(v2fDouble i) : SV_Target{ return tex2Dlod(texCopySource, float4(i.uv[0], 0, 0)); }
				ENDCG }

			// 13 - CalculateNormals
			Pass{ CGPROGRAM
				#pragma vertex vertSingle #pragma fragment frag
				half4 frag(v2fShed i) : SV_Target{ return half4(calculateNormal(i.uv), 1); }
				ENDCG }

			// 14 - TexCopyTemporalSPSR
			Pass{ CGPROGRAM
				#pragma vertex vertSPSRCopy #pragma fragment frag
				half4 frag(v2fSingle i) : SV_Target{ return tex2Dlod(temporalTexCopySource, float4(i.uv, 0, 0)); }
				ENDCG }
		}

}
	