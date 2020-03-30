// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Hidden/ReadNormal"
{
	Properties
	{
		_MainTex("", 2D) = "" {}
	}

		SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always
		//Tags{ "RenderType" = "Opaque" }
		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			sampler2D _CameraDepthTexture;
			sampler2D _CameraDepthNormalsTexture;



			float SampleDepth(float2 uv)
			{
				float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
					return DecodeFloatRG(cdn.zw) * _ProjectionParams.z;
			}

			float3 SampleNormal(float2 uv)
			{
				float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
					return DecodeViewNormalStereo(cdn);
			}

			float4 SampleDepthNormal(float2 uv)
			{
				float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
				float3 normal = DecodeViewNormalStereo(cdn);
				float depth = DecodeFloatRG(cdn.zw) * _ProjectionParams.z;
				return float4(normal, depth);
			}

			struct a2v {
				float4 vt:POSITION;
				float2 uv:TEXCOORD0;
			};
			struct v2f {
				float4 position:POSITION;
				float2 uv : TEXCOORD0;
			};


			//Vertex Shader  
			v2f vert(v2f v) {
				v.position = UnityObjectToClipPos(v.position);
				//v.position = UnityObjectToClipPos(v.position);
				return v;
			}

			//Fragment Shader  
			float4 frag(v2f i) : SV_TARGET{

				float4 midl = SampleDepthNormal(i.uv);
				//UNITY_SAMPLE_Normal(tex2D(_CameraDepthTexture, i.uv));
				//float depthValue = Linear01Depth(tex2Dproj(_CameraDepthTexture, UNITY_PROJ_COORD(i.scrPos)).r);

				//return half4(1.0, 0.0, 0.5, 1);
				return float4((midl.xyz+1.0)*0.5, 1.0);
				//float linearDepth = Linear01Depth(midl.w) * 100.0;
				//return half4(linearDepth, linearDepth, linearDepth, 1.0);
			}

			ENDCG
		}
	}
}
