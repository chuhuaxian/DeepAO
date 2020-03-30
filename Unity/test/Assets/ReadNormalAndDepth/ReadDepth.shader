// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Hidden/ReadDepth"
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

			//sampler2D _CameraDepthTexture;

			//struct v2f {
			//   float4 pos : SV_POSITION;
			//   float4 scrPos:TEXCOORD1;
			//};

			//v2f vert(appdata_base v) {
			//   v2f o;
			//   o.pos = UnityObjectToClipPos(v.vertex);
			//   o.scrPos = ComputeScreenPos(o.pos);
			//   return o;
			//}

			//float4 frag(v2f i) : COLOR
			//{
			//	float depthValue = Linear01Depth(tex2Dproj(_CameraDepthTexture, UNITY_PROJ_COORD(i.scrPos)).r); //将非线性变换到线性，范围前后都是0~1
			//	float4 depth;

			//	depth.r = 1.0;
			//	depth.g = depthValue*65535%255/255;
			//	depth.b = depthValue;
			//	depth.a = 1;
			//	return depth;
			//}

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
					return DecodeViewNormalStereo(cdn) * float3(1, 1, -1);
			}

			float4 SampleDepthNormal(float2 uv)
			{
				float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
					float3 normal = DecodeViewNormalStereo(cdn) * float3(1, 1, -1);
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

			float3 ReconstructViewPosition(float2 uv, float depth)
			{
				const float2 p11_22 = float2(unity_CameraProjection._11, unity_CameraProjection._22);
				const float2 p13_31 = float2(unity_CameraProjection._13, unity_CameraProjection._23);
				return float3((uv * 2.0 - 1.0 - p13_31) / p11_22 * depth, depth);
			}

			//Vertex Shader  
			v2f vert(v2f v) {
				v.position = UnityObjectToClipPos(v.position);
				return v;
			}

			//Fragment Shader  
			float4 frag(v2f i) : SV_TARGET{
				
				/*float midl = SampleDepth(i.uv);*/
				//float midl = UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, i.uv));

				//float3 base = ReconstructViewPosition(i.uv, midl);

				float depth = tex2D(_CameraDepthTexture, i.uv).r;
				//float depth = UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, i.uv));
				//float linearDepth = Linear01Depth(depth)*40.0;
				//float depth_z = Linear01Depth(base.z);
				//float depth_y = Linear01Depth(base.y);
				//float depth_x = Linear01Depth(base.z);

				depth = Linear01Depth(depth) * _ProjectionParams.z;

				float3 base = ReconstructViewPosition(i.uv, depth);

				/*float depth_y = Linear01Depth(base.y);
				float depth_x = Linear01Depth(base.x);*/
			/*	float depth_x = base.x ;
				float depth_y = base.y ;
				float depth_z = base.z;*/

				float depth_x = (base.x + 750.0) / _ProjectionParams.z;
				float depth_y = (base.y + 750.0) / _ProjectionParams.z;
				float depth_z = base.z / _ProjectionParams.z;
	
				//float depth_frac = depth * 65535 % 255 / 255;

				/*return LinearToGammaSpace(float3(depth_z, depth_z, depth_z));*/
				//return float4(depth_x, depth_y, 1-depth_z, 1.0);

				return float4(depth_x, depth_y, 1-depth_z, 1.0);
				//return float4(depth_x, depth_y, depth_z, 1.0);
				//return float4(depth_z * 65535 * 65535 % 255 / 255, depth_z * 65535 % 255 / 255, depth_z, 1);
				//return float4(1.0, depth_frac, depth, 1);
			}

			ENDCG
		}
	}
}
