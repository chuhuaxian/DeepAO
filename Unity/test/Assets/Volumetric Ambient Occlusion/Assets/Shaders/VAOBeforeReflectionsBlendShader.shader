// Copyright (c) 2016-2018 Jakub Boksansky - All Rights Reserved
// Volumetric Ambient Occlusion Unity Plugin 2.0

Shader "Hidden/Wilberforce/VAOBeforeReflectionsBlendShader"
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

			// 0 - BlendBeforeReflections
			Pass{Blend 0 Zero One, Zero SrcAlpha // Multiply destination alpha by source alpha
				Blend 1 DstColor Zero, Zero One // Multiplicative
				CGPROGRAM
				#pragma vertex vertDoubleTexCopy #pragma fragment frag
				AoOutput frag(v2fDouble i) { return blendBeforeReflections(i); }
				ENDCG }

			// 1 - BlendBeforeReflectionsLog
			Pass{Blend 0 Zero One, Zero SrcAlpha // Multiply destination alpha by source alpha
				Blend 1 One Zero // Overwrite
				CGPROGRAM
				#pragma vertex vertDoubleTexCopy #pragma fragment frag
				AoOutput frag(v2fDouble i) { return blendBeforeReflectionsLog(i); }
				ENDCG }
		}

		// Fallback for systems where "Blend N" is not supported
		SubShader
		{
			Cull Off ZWrite Off ZTest Always

			// 0 - BlendBeforeReflections
			Pass{CGPROGRAM
				#pragma vertex vertDoubleTexCopy #pragma fragment frag
				AoOutput frag(v2fDouble i) { return blendBeforeReflections(i); }
				ENDCG }

			// 1 - BlendBeforeReflectionsLog
			Pass{CGPROGRAM
				#pragma vertex vertDoubleTexCopy #pragma fragment frag
				AoOutput frag(v2fDouble i) { return blendBeforeReflectionsLog(i); }
				ENDCG }
		}
	}
