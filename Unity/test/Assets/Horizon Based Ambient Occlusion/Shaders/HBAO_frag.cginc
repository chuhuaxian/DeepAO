//#ifndef HBAO_FRAG_INCLUDED
//#define HBAO_FRAG_INCLUDED
//
//	inline float3 FetchViewPos(float2 uv) 
//	{
//		float z = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv));  // ----------------- DEPTH --------------------/
//		return float3((uv * _UVToView.xy + _UVToView.zw) * z, z);
//	}
//
//	float4 SampleDepthNormal(float2 uv)
//	{
//		float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
//		float3 normal = DecodeViewNormalStereo(cdn);
//		//normal = normal;
//		float depth = DecodeFloatRG(cdn.zw) * _ProjectionParams.z;
//		normal.yz = -normal.yz;
//		return float4(normal, depth);
//	}
//
//	float3 SampleNormal(float2 uv)
//	{
//		float4 cdn = tex2D(_CameraDepthNormalsTexture, uv);
//		return DecodeViewNormalStereo(cdn);
//	}
//
//
//
//	inline float3 FetchLayerViewPos(float2 uv) 
//	{
//		float z = SAMPLE_DEPTH_TEXTURE(_DepthTex, uv);
//		return float3((uv * _UVToView.xy + _UVToView.zw) * z, z);
//	}
//
//	inline float Falloff(float distanceSquare) 
//	{
//		// 1 scalar mad instruction
//		return distanceSquare * _NegInvRadius2 + 1.0;
//	}
//
//	inline float ComputeAO(float3 P, float3 N, float3 S) 
//	{
//		float3 V = S - P;
//		float VdotV = dot(V, V);
//		float NdotV = dot(N, V) * rsqrt(VdotV);
//
//		// Use saturate(x) instead of max(x,0.f) because that is faster on Kepler
//		return saturate(NdotV - _AngleBias) * saturate(Falloff(VdotV));
//	}
//
//	inline float3 MinDiff(float3 P, float3 Pr, float3 Pl) 
//	{
//		float3 V1 = Pr - P;
//		float3 V2 = P - Pl;
//		return (dot(V1, V1) < dot(V2, V2)) ? V1 : V2;
//	}
//
//	inline float2 RotateDirections(float2 dir, float2 rot) 
//	{
//		return float2(dir.x * rot.x - dir.y * rot.y,
//					  dir.x * rot.y + dir.y * rot.x);
//	}
//
//	//其中_ZBufferParams的定义如下：
//	static float my_FarClip = 1500.0;
//	static float my_NearClip = 0.01;
//	// OpenGL would be this:
//	//float zc0 = (1.0 - 1500.0 / 0.01) / 2.0;
//	//float zc1 = (1.0 + 1500.0 / 0.01) / 2.0;
//	//// D3D is this:
//	float zc0 = 1.0 - 1500.0 / 0.01;
//	float zc1 = 1500.0 / 0.01;
//	// now set _ZBufferParams with (zc0, zc1, zc0/m_FarClip, zc1/m_FarClip);
//	//float4 MyZBufferParams = float4(zc0, zc1, zc0 / m_FarClip, zc1 / m_FarClip);
//
//	// _ZBufferParams  Used to linearize Z buffer values. x is (1-far/near), y is (far/near), z is (x/far) and w is (y/far).
//	// Z buffer to linear 0..1 depth
//	inline float MyLinear01Depth(float z)
//	{
//		return 1.0 / (_ZBufferParams.x * z + _ZBufferParams.y);
//	}
//	// Z buffer to linear depth
//	inline float MyLinearEyeDepth(float z)
//	{
//		return 1.0 / (_ZBufferParams.z * z + _ZBufferParams.w);
//	}
//
//	float tanHalfFovY = 0.4663077;
//
//	float3 ReconstructViewPosition(float2 uv, float depth)
//	{
//		const float2 p11_22 = float2(unity_CameraProjection._11, unity_CameraProjection._22);
//		const float2 p13_31 = float2(unity_CameraProjection._13, unity_CameraProjection._23);
//		return float3((uv * 2.0 - 1.0 - p13_31) / p11_22 * depth, depth);
//	}
//
//	float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target 
//	{	
//		
//		float4 midl = SampleDepthNormal(i.uv2);
//		float3 N = midl.xyz;
//
//
//		float4 mydepthtex = tex2D(_MyDepthTex, i.uv2);
//		float mydepth = mydepthtex.r + mydepthtex.g / 255.0;
//		//float3 N = tex2D(_MyNormalTex, i.uv2).rgb;
//
//		//N = N * 2.0 - 1.0;
//		float4 MyUVToView = float4(2.0 * 0.4663077, -2.0 * 0.4663077, -1.0 * 0.4663077, 1.0 * 0.4663077);
//
//		float zz = MyLinearEyeDepth(UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, i.uv2)));
//		float3 P = float3((i.uv2 * MyUVToView.xy + MyUVToView.zw) * zz, zz);
//		//float3 P = ReconstructViewPosition(i.uv2, zz);
//		//P.y = - P.y;
//	
//
//		//_UVToView: (2.0f * invFocalLenX, -2.0f * invFocalLenY, -1.0f * invFocalLenX, 1.0f * invFocalLenY)
//		 	// tanHalfFovY = Mathf.Tan(0.5f * _hbaoCamera.fieldOfView * Mathf.Deg2Rad);
//
//		//float midll = UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, i.uv2));
//		//float3	P = ReconstructViewPosition(i.uv2, midll);
//		//P = float3(MyLinear01Depth(P.x), MyLinear01Depth(P.y), MyLinear01Depth(P.z));
//
//		//float depth_z = Linear01Depth(base.z);
//
//		//clip(_MaxDistance - P.z);
//
//		//float depth = midl.w;
//
//		float stepSize = min((_Radius / P.z), _MaxRadiusPixels) / (STEPS + 1.0);
//
//		float3 rand = tex2D(_NoiseTex, screenPos.xy / _NoiseTexSize).rgb;
//
//
//
//		float2 InvScreenParams = _ScreenParams.zw - 1.0;
//
//		const float alpha = 2.0 * UNITY_PI / DIRECTIONS;
//		float ao = 0;
//
//
//		for (int d = 0; d < DIRECTIONS; ++d) 
//		{
//			float angle = alpha * float(d);
//
//			// Compute normalized 2D direction
//			float cosA, sinA;
//			sincos(angle, sinA, cosA);
//			float2 direction = RotateDirections(float2(cosA, sinA), rand.xy);
//
//			// Jitter starting sample within the first step
//			float rayPixels = (rand.z * stepSize + 1.0);
//
//			for (int s = 0; s < STEPS; ++s) 
//			{
//				float2 snappedUV = round(rayPixels * direction) * InvScreenParams + i.uv2;
//				float3 S = FetchViewPos(snappedUV);
//				rayPixels += stepSize;
//				float contrib = ComputeAO(P, N, S);
//				ao += contrib;
//			}
//		}
//
//		ao *= (_AOmultiplier / (STEPS * DIRECTIONS));
//		float fallOffStart = _MaxDistance - _DistanceFalloff;
//		ao = lerp(saturate(1.0 - ao), 1.0, saturate((P.z - fallOffStart) / (_MaxDistance - fallOffStart)));
//
//		//return float4(mynormal.rgb, 1.0);
//		return float4(ao, ao, ao, 1.0);
//		//return float4(mynormaltex.rgb, 1.0);
//		//return float4(N, 1.0);
//		//return float4(midl.xyz, 1.0);
//		//return float4(0.1, 0.3, 0.5, N.x);
//		//return float4(midll, midll, midll, 1.0);
//
//	}
//
//#endif // HBAO_FRAG_INCLUDED


	//----------------------------------------------------------------------------------
	//
	// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
	//
	// Redistribution and use in source and binary forms, with or without
	// modification, are permitted provided that the following conditions
	// are met:
	//  * Redistributions of source code must retain the above copyright
	//    notice, this list of conditions and the following disclaimer.
	//  * Redistributions in binary form must reproduce the above copyright
	//    notice, this list of conditions and the following disclaimer in the
	//    documentation and/or other materials provided with the distribution.
	//  * Neither the name of NVIDIA CORPORATION nor the names of its
	//    contributors may be used to endorse or promote products derived
	//    from this software without specific prior written permission.
	//
	// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
	// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
	// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
	// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
	// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
	// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
	// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	//
	//----------------------------------------------------------------------------------

#ifndef HBAO_FRAG_INCLUDED
#define HBAO_FRAG_INCLUDED

	inline float3 FetchViewPos(float2 uv)
	{
		// References: https://docs.unity3d.com/Manual/SL-PlatformDifferences.html

#ifdef UNITY_SINGLE_PASS_STEREO  // False
		float2 uvDepth = UnityStereoScreenSpaceUVAdjust(uv, _CameraDepthTexture_ST) * _TargetScale.xy;
#else
		float2 uvDepth = uv * _TargetScale.xy;
#endif // UNITY_SINGLE_PASS_STEREO


#if ORTHOGRAPHIC_PROJECTION_ON  // False
		float z = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uvDepth);

#if defined(UNITY_REVERSED_Z)
		z = 1 - z;
#endif // UNITY_REVERSED_Z
		z = _ProjectionParams.y + z * (_ProjectionParams.z - _ProjectionParams.y); // near + depth * (far - near)
#else  // execute this
		float z = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uvDepth));  // ----------------- DEPTH --------------------//
#endif // ORTHOGRAPHIC_PROJECTION_ON

		return float3((uv * _UVToView.xy + _UVToView.zw) * z, z);
	}

	inline float3 FetchLayerViewPos(float2 uv)
	{
		float z = SAMPLE_DEPTH_TEXTURE(_DepthTex, uv);
		return float3((uv * _UVToView.xy + _UVToView.zw) * z, z);
	}

	inline float Falloff(float distanceSquare)
	{
		// 1 scalar mad instruction
		return distanceSquare * _NegInvRadius2 + 1.0;
	}

	inline float ComputeAO(float3 P, float3 N, float3 S)
	{
		float3 V = S - P;
		float VdotV = dot(V, V);
		float NdotV = dot(N, V) * rsqrt(VdotV);

		// Use saturate(x) instead of max(x,0.f) because that is faster on Kepler
		return saturate(NdotV - _AngleBias) * saturate(Falloff(VdotV));
	}

	inline float3 MinDiff(float3 P, float3 Pr, float3 Pl)
	{
		float3 V1 = Pr - P;
		float3 V2 = P - Pl;
		return (dot(V1, V1) < dot(V2, V2)) ? V1 : V2;
	}

	inline float2 RotateDirections(float2 dir, float2 rot)
	{
		return float2(dir.x * rot.x - dir.y * rot.y,
			dir.x * rot.y + dir.y * rot.x);
	}

#if COLOR_BLEEDING_ON  // True
	static float2 cbUVs[DIRECTIONS * STEPS];
	static float cbContribs[DIRECTIONS * STEPS];
#endif

	half4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
	{
#if DEINTERLEAVED  // False
		float3 P = FetchLayerViewPos(i.uv2);
#else  // ====>this branch
		float3 P = FetchViewPos(i.uv2);  // ----------------- DEPTH --------------------//
#endif  // DEINTERLEAVED

		clip(_MaxDistance - P.z);
		float stepSize = min((_Radius / P.z), _MaxRadiusPixels) / (STEPS + 1.0);

#if DEINTERLEAVED  // False
		// (cos(alpha), sin(alpha), jitter)
		float3 rand = _Jitter.xyz;
		float3 N = tex2D(_NormalsTex, i.uv2).rgb * 2.0 - 1.0;  // ------------------------ NORMAL --------------------------//
#else  // ====>this branch
		// (cos(alpha), sin(alpha), jitter)
		float3 rand = tex2D(_NoiseTex, screenPos.xy / _NoiseTexSize).rgb;
		float2 InvScreenParams = _ScreenParams.zw - 1.0;

#if NORMALS_RECONSTRUCT  // False
		float3 Pr, Pl, Pt, Pb;
		Pr = FetchViewPos(i.uv2 + float2(InvScreenParams.x, 0));
		Pl = FetchViewPos(i.uv2 + float2(-InvScreenParams.x, 0));
		Pt = FetchViewPos(i.uv2 + float2(0, InvScreenParams.y));
		Pb = FetchViewPos(i.uv2 + float2(0, -InvScreenParams.y));
		float3 N = normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));
#else  // ====>this branch
#if NORMALS_CAMERA // False
#if UNITY_SINGLE_PASS_STEREO  // False
		float3 N = DecodeViewNormalStereo(tex2D(_CameraDepthNormalsTexture, UnityStereoScreenSpaceUVAdjust(i.uv2, _CameraDepthTexture_ST)));
#else  
		float3 N = DecodeViewNormalStereo(tex2D(_CameraDepthNormalsTexture, i.uv2));
#endif // UNITY_SINGLE_PASS_STEREO
#else	// ====>this branch
#if UNITY_SINGLE_PASS_STEREO  // False
		float3 N = tex2D(_CameraGBufferTexture2, UnityStereoScreenSpaceUVAdjust(i.uv2, _CameraDepthTexture_ST)).rgb * 2.0 - 1.0;
#else
		float3 N = tex2D(_CameraGBufferTexture2, i.uv2).rgb * 2.0 - 1.0;
#endif // UNITY_SINGLE_PASS_STEREO
		N = mul((float3x3)_WorldToCameraMatrix, N);
#endif // NORMALS_CAMERA
		N = float3(N.x, -N.yz);
#endif // NORMALS_RECONSTRUCT
#endif // DEINTERLEAVED


		const float alpha = 2.0 * UNITY_PI / DIRECTIONS;
		float ao = 0;

		UNITY_UNROLL
			for (int d = 0; d < DIRECTIONS; ++d)
			{
				float angle = alpha * float(d);

				// Compute normalized 2D direction
				float cosA, sinA;
				sincos(angle, sinA, cosA);
				float2 direction = RotateDirections(float2(cosA, sinA), rand.xy);

				// Jitter starting sample within the first step
				float rayPixels = (rand.z * stepSize + 1.0);

				UNITY_UNROLL
					for (int s = 0; s < STEPS; ++s)
					{
#if DEINTERLEAVED  // False
						float2 snappedUV = round(rayPixels * direction) * _LayerRes_TexelSize.xy + i.uv2;
						float3 S = FetchLayerViewPos(snappedUV);
#else  // ====>this branch
						float2 snappedUV = round(rayPixels * direction) * InvScreenParams + i.uv2;
						float3 S = FetchViewPos(snappedUV);
#endif // DEINTERLEAVED

						rayPixels += stepSize;

						float contrib = ComputeAO(P, N, S);

#if OFFSCREEN_SAMPLES_CONTRIB
						float2 offscreenAmount = _OffscreenSamplesContrib * (snappedUV - saturate(snappedUV) != 0 ? 1 : 0);
						contrib = max(contrib, offscreenAmount.x);
						contrib = max(contrib, offscreenAmount.y);
#endif  // OFFSCREEN_SAMPLES_CONTRIB

						ao += contrib;

#if COLOR_BLEEDING_ON  // True
						int sampleIdx = d * s;
						cbUVs[sampleIdx] = snappedUV;
						cbContribs[sampleIdx] = contrib;
#endif  // COLOR_BLEEDING_ON
					}
			}

		ao *= (_AOmultiplier / (STEPS * DIRECTIONS));
		float fallOffStart = _MaxDistance - _DistanceFalloff;
		ao = lerp(saturate(1.0 - ao), 1.0, saturate((P.z - fallOffStart) / (_MaxDistance - fallOffStart)));

#if COLOR_BLEEDING_ON  // True
		half3 col = half3(0.0, 0.0, 0.0);
		UNITY_UNROLL
			for (int s = 0; s < DIRECTIONS * STEPS; s += 2)
			{
#if UNITY_SINGLE_PASS_STEREO  // False
				float2 uvCB = UnityStereoScreenSpaceUVAdjust(float2(cbUVs[s].x, cbUVs[s].y * _MainTex_TexelSize.y * _MainTex_TexelSize.w), _MainTex_ST);
#else
				float2 uvCB = float2(cbUVs[s].x, cbUVs[s].y * _MainTex_TexelSize.y * _MainTex_TexelSize.w);
#endif // UNITY_SINGLE_PASS_STEREO

				half3 emission = tex2D(_MainTex, uvCB).rgb;
				half average = (emission.x + emission.y + emission.z) / 3;
				half scaledAverage = saturate((average - _ColorBleedBrightnessMaskRange.x) / (_ColorBleedBrightnessMaskRange.y - _ColorBleedBrightnessMaskRange.x + 1e-6));
				half maskMultiplier = 1 - (scaledAverage * _ColorBleedBrightnessMask);
				col += emission * cbContribs[s] * maskMultiplier;
			}
		col /= DIRECTIONS * STEPS;
#if DEFERRED_SHADING_ON
#if UNITY_SINGLE_PASS_STEREO
		half3 albedo = tex2D(_CameraGBufferTexture0, UnityStereoScreenSpaceUVAdjust(i.uv2, _MainTex_ST)).rgb * 0.8 + 0.2;
#else
		half3 albedo = tex2D(_CameraGBufferTexture0, i.uv2).rgb * 0.8 + 0.2;
#endif // UNITY_SINGLE_PASS_STEREO
		col = saturate(1 - lerp(dot(col, 0.333).xxx, col * _AlbedoMultiplier * albedo, _ColorBleedSaturation));
#else
		col = saturate(1 - lerp(dot(col, 0.333).xxx, col, _ColorBleedSaturation));
#endif  // DEFERRED_SHADING_ON
#else
		half3 col = half3(EncodeFloatRG(saturate(P.z * (1.0 / _ProjectionParams.z))), 1.0);
#endif  //COLOR_BLEEDING_ON

		return half4(ao, ao, ao, 1.0);
	}

#endif // HBAO_FRAG_INCLUDED

