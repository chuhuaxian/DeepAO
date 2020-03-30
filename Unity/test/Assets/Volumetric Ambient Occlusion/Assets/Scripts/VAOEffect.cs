// Copyright (c) 2016-2018 Jakub Boksansky - All Rights Reserved
// Volumetric Ambient Occlusion Unity Plugin 2.0

using UnityEngine;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.AnimatedValues;
#endif
using System;
using System.Collections.Generic;
using System.Reflection;


namespace Wilberforce.VAO
{

    [ExecuteInEditMode]
    [RequireComponent(typeof(Camera))]
    [HelpURL("https://projectwilberforce.github.io/vaomanual/")]
    [AddComponentMenu("Image Effects/Rendering/Volumetric Ambient Occlusion")]
    public class VAOEffect : VAOEffectCommandBuffer
    {
        [ImageEffectOpaque]
        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            this.PerformOnRenderImage(source, destination);
        }

       


    }


#if UNITY_EDITOR

    [CustomEditor(typeof(VAOEffect))]
    public class VAOEffectEditorImageEffect : VAOEffectEditor { }

#endif
}
