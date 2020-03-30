// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Hidden/GetInput"
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

			float3 ReconstructViewPosition(float2 uv, float depth)
			{
				const float2 p11_22 = float2(unity_CameraProjection._11, unity_CameraProjection._22);
				const float2 p13_31 = float2(unity_CameraProjection._13, unity_CameraProjection._23);
				return float3((uv * 2 - 1 - p13_31) / p11_22 * depth, depth);
			}

			//Fragment Shader  
			float4 frag(v2f i) : SV_TARGET{

				float3 normal = SampleNormal(i.uv);
				normal = (normal + 1.0)*0.5;
				//UNITY_SAMPLE_Normal(tex2D(_CameraDepthTexture, i.uv));
				//float depthValue = Linear01Depth(tex2Dproj(_CameraDepthTexture, UNITY_PROJ_COORD(i.scrPos)).r);

				float midl = UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, i.uv));

				float3 base = ReconstructViewPosition(i.uv, midl);
				//float depth = UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, i.uv));
				//float linearDepth = Linear01Depth(depth)*40.0;
				//float depth_z = Linear01Depth(base.z);
				//float depth_y = Linear01Depth(base.y);
				//float depth_x = Linear01Depth(base.z);

				float depth_z = Linear01Depth(base.z);

				//return half4(1.0, 0.0, 0.5, 1);
				//return float4(depth_z, depth_z, depth_z, 1.0);
				return float4(normal, 1-depth_z);
				//float linearDepth = Linear01Depth(midl.w) * 100.0;
				//return half4(linearDepth, linearDepth, linearDepth, 1.0);
			}

			ENDCG
		}
	}
}
