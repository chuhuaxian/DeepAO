#ifndef _AREA_LIGHT_
#define _AREA_LIGHT_

#include "ShadingModel.hlsl"

#define SHARP_EDGE_FIX 1
#define WITHOUT_CORRECT_HORIZON 0
#define WITH_GG_Sphere 1

float GetSphereLight(float radiusTan, float NoL, float NoV, float VoL)
{
    // radiusCos can be precalculated if radiusTan is a directional light
    float radiusCos = rsqrt(1 + pow2(radiusTan));
    
    // Early out if R falls within the disc
    float RoL = 2 * NoL * NoV - VoL;
    if (RoL >= radiusCos)
        return 1;

    float rOverLengthT = radiusCos * radiusTan * rsqrt(1 - RoL * RoL);
    float NoTr = rOverLengthT * (NoV - RoL * NoL);
    float VoTr = rOverLengthT * (2 * NoV * NoV - 1 - RoL * VoL);

#if WITH_GG_Sphere
    // Calculate dot(cross(N, L), V). This could already be calculated and available.
    float triple = sqrt(saturate(1 - NoL * NoL - NoV * NoV - VoL * VoL + 2 * NoL * NoV * VoL));
    // Do one Newton iteration to improve the bent light vector
    float NoBr = rOverLengthT * triple, VoBr = rOverLengthT * (2 * triple * NoV);
    float NoLVTr = NoL * radiusCos + NoV + NoTr, VoLVTr = VoL * radiusCos + 1 + VoTr;
    float p = NoBr * VoLVTr, q = NoLVTr * VoLVTr, s = VoBr * NoLVTr;    
    float xNum = q * (-0.5 * p + 0.25 * VoBr * NoLVTr);
    float xDenom = p * p + s * ((s - 2 * p)) + NoLVTr * ((NoL * radiusCos + NoV) * VoLVTr * VoLVTr + q * (-0.5 * (VoLVTr + VoL * radiusCos) - 0.5));
    float twoX1 = 2 * xNum / (xDenom * xDenom + xNum * xNum);
    float sinTheta = twoX1 * xDenom;
    float cosTheta = 1 - twoX1 * xNum;
    NoTr = cosTheta * NoTr + sinTheta * NoBr; // use new T to update NoTr
    VoTr = cosTheta * VoTr + sinTheta * VoBr; // use new T to update VoTr
#endif

    // Calculate (N.H)^2 based on the bent light vector
    float newNoL = NoL * radiusCos + NoTr;
    float newVoL = VoL * radiusCos + VoTr;
    float NoH = NoV + newNoL;
    float HoH = 2 * newVoL + 2;
    return max(0, NoH * NoH / HoH);
}

/*
//Init_Sphere( LightData, saturate( lightRadius * rsqrt( dot(_LightPos.rgb - worldPos, _LightPos.rgb - worldPos) ) * (1 - Pow2(Roughness) ) ) );
void Init_Sphere(inout BSDFContext Context, float SinAlpha)
{
    if (SinAlpha > 0)
    {
        float CosAlpha = sqrt(1 - Pow2(SinAlpha));

        float RoL = 2 * Context.NoL * Context.NoV - Context.VoL;
        if (RoL >= CosAlpha)
        {
            Context.NoH = 1;
            Context.VoH = abs(Context.NoV);
        }
        else
        {
            float rInvLengthT = SinAlpha * rsqrt(1 - RoL * RoL);
            float NoTr = rInvLengthT * (Context.NoV - RoL * Context.NoL);
            float VoTr = rInvLengthT * (2 * Context.NoV * Context.NoV - 1 - RoL * Context.VoL);

#if WITH_GG_Sphere
                // dot( cross(N,L), V )
                float NxLoV = sqrt(saturate(1 - Pow2(Context.NoL) - Pow2(Context.NoV) - Pow2(Context.VoL) + 2 * Context.NoL * Context.NoV * Context.VoL));

                float NoBr = rInvLengthT * NxLoV;
                float VoBr = rInvLengthT * NxLoV * 2 * Context.NoV;
                float NoLVTr = Context.NoL * CosAlpha + Context.NoV + NoTr;
                float VoLVTr = Context.VoL * CosAlpha + 1 + VoTr;

                float p = NoBr * VoLVTr;
                float q = NoLVTr * VoLVTr;
                float s = VoBr * NoLVTr;

                float xNum = q * (-0.5 * p + 0.25 * VoBr * NoLVTr);
                float xDenom = p * p + s * (s - 2 * p) + NoLVTr * ((Context.NoL * CosAlpha + Context.NoV) * Pow2(VoLVTr) + q * (-0.5 * (VoLVTr + Context.VoL * CosAlpha) - 0.5));
                float TwoX1 = 2 * xNum / (Pow2(xDenom) + Pow2(xNum));
                float SinTheta = TwoX1 * xDenom;
                float CosTheta = 1.0 - TwoX1 * xNum;
                NoTr = CosTheta * NoTr + SinTheta * NoBr;
                VoTr = CosTheta * VoTr + SinTheta * VoBr;
#endif

            Context.NoL = Context.NoL * CosAlpha + NoTr;
            Context.VoL = Context.VoL * CosAlpha + VoTr;
            float InvLenH = rsqrt(2 + 2 * Context.VoL);
            Context.NoH = saturate((Context.NoL + Context.NoV) * InvLenH);
            Context.VoH = saturate(InvLenH + InvLenH * Context.VoL);
        }
    }
}
*/

void AreaLightIntegrated(float3 pos, float3 tubeStart, float3 tubeEnd, float3 normal, float tubeRad, float3 representativeDir, out float3 outLightDir, out float outNdotL, out half outLightDist)
{
    half3 N = normal;
    float3 L0 = tubeStart - pos;
    float3 L1 = tubeEnd - pos;
    float L0dotL0 = dot(L0, L0);
    float distL0 = sqrt(L0dotL0);
    float distL1 = length(L1);

    float NdotL0 = dot(L0, N) / (2 * distL0);
    float NdotL1 = dot(L1, N) / (2 * distL1);
    outNdotL = saturate(NdotL0 + NdotL1);

    float3 Ldir = L1 - L0;
    float RepdotL0 = dot(representativeDir, L0);
    float RepdotLdir = dot(representativeDir, Ldir);
    float L0dotLdir = dot(L0, Ldir);
    float LdirdotLdir = dot(Ldir, Ldir);
    float distLdir = sqrt(LdirdotLdir);

#if SHARP_EDGE_FIX
    float t = (L0dotLdir * RepdotL0 - L0dotL0 * RepdotLdir) / (L0dotLdir * RepdotLdir - LdirdotLdir * RepdotL0);
    t = saturate(t);

    float3 L0xLdir = cross(L0, Ldir);
    float3 LdirxR = cross(Ldir, representativeDir);
    float RepAtLdir = dot(L0xLdir, LdirxR);

    t = lerp(1 - t, t, step(0, RepAtLdir));

#else
    float t = (RepdotL0 * RepdotLdir - L0dotLdir) / (distLdir * distLdir - RepdotLdir * RepdotLdir);
    t = saturate(t);

#endif

    float3 closestPoint = L0 + Ldir * t;
    float3 centerToRay = dot(closestPoint, representativeDir) * representativeDir - closestPoint;

    closestPoint = closestPoint + centerToRay * saturate(tubeRad / length(centerToRay));

    outLightDist = length(closestPoint);
    outLightDir = closestPoint / outLightDist;
}


/////////////////////////////////////////////////////////////////////////***Falloff***/////////////////////////////////////////////////////////////////////////
half GetLumianceIntensity(half lumiance)
{
    return max(0, lumiance) / (4 * PI);
}

half SmoothFalloff(half squaredDistance, half invSqrAttRadius)
{
    half factor = squaredDistance * invSqrAttRadius;
    half smoothFactor = saturate(1 - factor * factor);
    return smoothFactor * smoothFactor;
}
half DistanceFalloff(half3 unLightDir, half invSqrAttRadius)
{
    half sqrDist = dot(unLightDir, unLightDir);
    half attenuation = 1 / (max(sqrDist, 0.01 * 0.01));
    attenuation *= SmoothFalloff(sqrDist, invSqrAttRadius);
    return attenuation;
}

half AngleFalloff(half3 normalizedLightVector, half3 lightDir, half lightAngleScale, half lightAngleOffset)
{
    // On the CPU
    // half lightAngleScale = 1 / max ( 0.001, (cosInner - cosOuter) );
    // half lightAngleOffset = -cosOuter * lightAngleScale ;

    half cd = dot(lightDir, normalizedLightVector);
    half attenuation = saturate(cd * lightAngleScale + lightAngleOffset);
    attenuation *= attenuation;
    return attenuation;
}

/*
half IESFalloff(half3 L)
{
    half3 iesSampleDirection = mul (light  worldToLight , -L);

    // Cartesian to spherical
    // Texture encoded with cos(phi), scale from -1 - >1 to 0 - >1
    half phiCoord = ( iesSampleDirection.z * 0.5) + 0.5;
    half theta = atan2 ( iesSampleDirection.y , iesSampleDirection.x);
    half thetaCoord = theta * Inv_Two_PI ;
    half3 texCoord = half3 (thetaCoord , phiCoord);
    half iesProfileScale = iesTexture . SampleLevel (sampler , texCoord , 0).r;
    return iesProfileScale ;
}
*/

/////////////////////////////////////////////////////////////////////////***Energy***/////////////////////////////////////////////////////////////////////////
//////Punctual Energy
half3 Point_Energy(half3 worldPos, half3 lightPos, half3 lightColor, half lumiance, half range, half NoL)
{
    half3 unLightDir = lightPos - worldPos ;
    half3 L = normalize(unLightDir);
    half Falloff = DistanceFalloff(unLightDir, (1 / (range * 10)));

    // lightColor is the outgoing luminance of the light time the user light color
    // i.e with point light and luminous power unit : lightColor = color * phi / (4 * PI)
    half3 luminance = Falloff * NoL * ( lightColor * GetLumianceIntensity(lumiance) );
    return luminance;
}

half3 Spot_Energy(half3 worldPos, half3 lightPos, half3 lightColor, half lumiance, half range, half NoL)
{
    half3 unLightDir = lightPos - worldPos ;
    half3 L = normalize(unLightDir);
    half Falloff = DistanceFalloff(unLightDir, (1 / (range * 10)));

    ///Falloff *= AngleFalloff(L, lightForward, lightAngleScale, lightAngleOffset);
    //half lightAngleScale = 1 / max ( 0.001, (90 - 30) );
    //half lightAngleOffset = -30 * lightAngleScale ;
    //Falloff *= AngleFalloff(L, half3(0, 90, 0), lightAngleScale, lightAngleOffset);

    // lightColor is the outgoing luminance of the light time the user light color
    // i.e with point light and luminous power unit : lightColor = color * phi / (4 * PI)
    half3 luminance = Falloff * NoL * ( lightColor * GetLumianceIntensity(lumiance) );
    return luminance;
}



//////Area Energy
half Sphere_Energy(half3 worldNormal, half3 worldPos, half3 lightPos, half3 lightColor, half radius, half range, half lumiance)
{
    half3 unLightDir = lightPos - worldPos;
    half3 L = normalize(unLightDir);
    half sqrDist = dot (unLightDir , unLightDir);
    half illuminance = 0;

#if WITHOUT_CORRECT_HORIZON // Analytical solution above horizon

    // Patch to Sphere frontal equation ( Quilez version )
    half sqrLightRadius = radius * radius;
    // Do not allow object to penetrate the light ( max )
    // Form factor equation include a (1 / PI ) that need to be cancel
    // thus the " PI *"
    illuminance = PI * (sqrLightRadius / (max(sqrLightRadius , sqrDist))) * saturate (dot(worldNormal , L));

#else // Analytical solution with horizon

    // Tilted patch to sphere equation
    half Beta = acos(saturate(dot(worldNormal, L)));
    half H = sqrt (sqrDist);
    half h = H / radius;
    half x = sqrt (h * h - 1);
    half y = -x * (1 / tan (Beta));

    if (h * cos (Beta) > 1) {
        illuminance = cos ( Beta ) / (h * h);
    } else {
        illuminance = (1 / (PI * h * h)) * (cos(Beta) * acos (y) - x * sin(Beta) * sqrt (1 - y * y)) + (1 / PI) * atan (sin (Beta) * sqrt (1 - y * y) / x);
    }
    illuminance *= PI;

#endif

    half RangeFalloff = DistanceFalloff(unLightDir, (1 / (range * 10)));
    half LumiancePower = lightColor * GetLumianceIntensity(lumiance);
    return illuminance * RangeFalloff * LumiancePower;
}

#endif