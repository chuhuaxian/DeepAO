// Copyright (c) 2016-2018 Jakub Boksansky - All Rights Reserved
// Volumetric Ambient Occlusion Unity Plugin 2.0

Shader "Hidden/Wilberforce/VAOStandardColorbleedShader"
{
	Properties
	{
		_MainTex("Texture", 2D) = "white" {}
	}
	CGINCLUDE

	#pragma target 3.0

	#define WFORCE_VAO_COLORBLEED_ON
	#pragma multi_compile DISABLE_TEMPORAL_ACCUMULATION ENABLE_TEMPORAL_ACCUMULATION	

	#include "VAO.cginc"

	ENDCG
	SubShader
	{
		Cull Off ZWrite Off ZTest Always

		// 0 - Culling prepass VAO
		Pass{ CGPROGRAM
			#pragma vertex vertShed #pragma fragment frag
			WFORCE_VAO_MAIN_PASS_RETURN_TYPE frag(v2fShed i) : SV_Target{ return ao(i, WFORCE_STANDARD_VAO, 0, 0, 0, 0, 1, 0, 0, 4); }
			ENDCG }

		// 1 - Main pass VAO single radius
		Pass{ CGPROGRAM
			#pragma vertex vertShed #pragma fragment frag
			WFORCE_VAO_MAIN_PASS_RETURN_TYPE frag(v2fShed i) : SV_Target{ return ao(i, WFORCE_STANDARD_VAO, 0, 0, 0, 0, 0, cullingPrepassMode, adaptiveMode, sampleCount); }
			ENDCG }

		// 2 - Main pass VAO double radius
		Pass{ CGPROGRAM
			#pragma vertex vertShed #pragma fragment frag
			WFORCE_VAO_MAIN_PASS_RETURN_TYPE frag(v2fShed i) : SV_Target{ return ao(i, WFORCE_STANDARD_VAO, 0, 0, 1, 4, 0, cullingPrepassMode, adaptiveMode, sampleCount); }
			ENDCG }

		// 3 - Main pass VAO triple radius
		Pass{ CGPROGRAM
			#pragma vertex vertShed #pragma fragment frag
			WFORCE_VAO_MAIN_PASS_RETURN_TYPE frag(v2fShed i) : SV_Target{ return ao(i, WFORCE_STANDARD_VAO, 1, 2, 1, 4, 0, cullingPrepassMode, adaptiveMode, sampleCount); }
			ENDCG }

		// 4 - Main pass VAO double radius HQ
		Pass{ CGPROGRAM
			#pragma vertex vertShed #pragma fragment frag
			WFORCE_VAO_MAIN_PASS_RETURN_TYPE frag(v2fShed i) : SV_Target{ return ao(i, WFORCE_STANDARD_VAO, 0, 0, 1, 8, 0, cullingPrepassMode, adaptiveMode, sampleCount); }
			ENDCG }

		// 5 - Main pass VAO triple radius HQ
		Pass{ CGPROGRAM
			#pragma vertex vertShed #pragma fragment frag
			WFORCE_VAO_MAIN_PASS_RETURN_TYPE frag(v2fShed i) : SV_Target{ return ao(i, WFORCE_STANDARD_VAO, 1, 4, 1, 8, 0, cullingPrepassMode, adaptiveMode, sampleCount); }
			ENDCG }
	}

}

