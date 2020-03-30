using UnityEngine;
using UnityEditor;
#if UNITY_5_5_OR_NEWER
using UnityEditor.Rendering;
#endif
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

public class HBAO_Core_Editor : Editor
{
    private HBAO_Core m_HBAO;
    private Texture2D m_HBAOTex;
    private GUIStyle m_SettingsGroupStyle;
    private GUIStyle m_TitleLabelStyle;
    private int m_SelectedPreset;
    // settings group <setting, property reference>
    private Dictionary<FieldInfo, List<SerializedProperty>> m_GroupFields = new Dictionary<FieldInfo, List<SerializedProperty>>();
    private readonly Dictionary<int, HBAO_Core.Preset> m_Presets = new Dictionary<int, HBAO_Core.Preset>()
    {
        { 0, HBAO_Core.Preset.Normal },
        { 1, HBAO_Core.Preset.FastPerformance },
        { 2, HBAO_Core.Preset.FastestPerformance },
        { 3, HBAO_Core.Preset.Custom },
        { 4, HBAO_Core.Preset.HighQuality },
        { 5, HBAO_Core.Preset.HighestQuality }
    };


    void OnEnable()
    {
        m_HBAO = (HBAO_Core)target;
        m_HBAOTex = Resources.Load<Texture2D>("hbao");

        var settingsGroups = typeof(HBAO_Core).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .Where(x => x.GetCustomAttributes(typeof(HBAO_Core.SettingsGroup), false).Any());
        foreach (var group in settingsGroups)
        {
            foreach (var setting in group.FieldType.GetFields(BindingFlags.Instance | BindingFlags.Public))
            {
                if (!m_GroupFields.ContainsKey(group))
                    m_GroupFields[group] = new List<SerializedProperty>();

                var property = serializedObject.FindProperty(group.Name + "." + setting.Name);
                if (property != null)
                    m_GroupFields[group].Add(property);
            }
        }

        m_SelectedPreset = m_Presets.Values.ToList().IndexOf(m_HBAO.presets.preset);
    }

    public override void OnInspectorGUI()
    {
        DrawGUI();
    }

    protected void DrawGUI(bool isIntegrated = false)
    {
        serializedObject.Update();

        SetStyles();

        EditorGUILayout.BeginVertical();
        {
            // header
            GUILayout.Space(10.0f);
            GUILayout.Label(m_HBAOTex, m_TitleLabelStyle, GUILayout.ExpandWidth(true));

            //if (m_HBAO.GetComponents<MonoBehaviour>()[0] != m_HBAO)
            //{
            //GUILayout.Space(6.0f);
            //EditorGUILayout.HelpBox("This Post FX should be one of the first in your effect stack", MessageType.Info);
            //}

            Event e = Event.current;

            // settings groups
            foreach (var group in m_GroupFields)
            {
                var groupProperty = serializedObject.FindProperty(group.Key.Name);
                if (groupProperty == null)
                    continue;

                GUILayout.Space(6.0f);
                Rect rect = GUILayoutUtility.GetRect(16f, 22f, m_SettingsGroupStyle);
                GUI.Box(rect, ObjectNames.NicifyVariableName(groupProperty.displayName), m_SettingsGroupStyle);
                if (e.type == EventType.MouseDown && rect.Contains(e.mousePosition))
                {
                    groupProperty.isExpanded = !groupProperty.isExpanded;
                    e.Use();
                }

                if (!groupProperty.isExpanded)
                    continue;

                // presets is a special case
                if (group.Key.FieldType == typeof(HBAO_Core.Presets))
                {
                    GUILayout.Space(6.0f);
                    m_SelectedPreset = GUILayout.SelectionGrid(m_SelectedPreset, m_Presets.Values.Select(x => ObjectNames.NicifyVariableName(x.ToString())).ToArray(), 3);
                    GUILayout.Space(6.0f);
                    if (GUILayout.Button("Apply Preset"))
                    {
                        m_HBAO.ApplyPreset(m_Presets[m_SelectedPreset]);
                        EditorUtility.SetDirty(target);
                        if (!EditorApplication.isPlaying)
                        {
#if (UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
                            EditorApplication.MarkSceneDirty();
#else
                            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
#endif
                        }
                    }
                }

                foreach (var field in group.Value)
                {
                    // hide real presets
                    if (group.Key.FieldType == typeof(HBAO_Core.Presets))
                        continue;

                    // hide integrationStage for legacy HBAO
                    if (group.Key.FieldType == typeof(HBAO_Core.GeneralSettings) && field.name == "integrationStage")
                    {
                        if (!isIntegrated)
                        {
                            continue;
                        }
                    }

                    // hide resolution when deinterleaved HBAO is on
                    if (group.Key.FieldType == typeof(HBAO_Core.GeneralSettings) && field.name == "resolution")
                    {
                        if (m_HBAO.generalSettings.deinterleaving != HBAO_Core.Deinterleaving.Disabled)
                        {
                            continue;
                        }
                    }
#if UNITY_5_4
                    // warn about deinterleaving not supported with SPSR
                    else if (group.Key.FieldType == typeof(HBAO_Core.GeneralSettings) && field.name == "displayMode")
                    {
                        if (m_HBAO.generalSettings.deinterleaving != HBAO_Core.Deinterleaving.Disabled &&
                            PlayerSettings.virtualRealitySupported && PlayerSettings.singlePassStereoRendering)
                        {
                            GUILayout.Space(6.0f);
                            EditorGUILayout.HelpBox("Deinterleaving is not supported with Single Pass Stereo Rendering...", MessageType.Warning);
                        }
                    }
#elif UNITY_5_5_OR_NEWER
                    // warn about deinterleaving not supported with SPSR
                    else if (group.Key.FieldType == typeof(HBAO_Core.GeneralSettings) && field.name == "displayMode")
                    {
                        if (m_HBAO.generalSettings.deinterleaving != HBAO_Core.Deinterleaving.Disabled &&
                            PlayerSettings.virtualRealitySupported && PlayerSettings.stereoRenderingPath == StereoRenderingPath.SinglePass)
                        {
                            GUILayout.Space(6.0f);
                            EditorGUILayout.HelpBox("Deinterleaving is not supported with Single Pass Stereo Rendering...", MessageType.Warning);
                        }
                    }
#endif
                    // hide noise type when deinterleaved HBAO is on
                    else if (group.Key.FieldType == typeof(HBAO_Core.GeneralSettings) && field.name == "noiseType")
                    {
                        if (m_HBAO.generalSettings.deinterleaving != HBAO_Core.Deinterleaving.Disabled)
                        {
                            continue;
                        }
                    }
#if !(UNITY_5_1 || UNITY_5_0)
                    // hide useMultiBounce setting in BeforeReflections integration stage
                    else if (group.Key.FieldType == typeof(HBAO_Core.AOSettings) && field.name == "useMultiBounce")
                    {
                        if (m_HBAO.generalSettings.integrationStage == HBAO_Core.IntegrationStage.BeforeReflections)
                        {
                            continue;
                        }
                    }
#endif
                    // hide multiBounceInfluence setting when not used
                    else if (group.Key.FieldType == typeof(HBAO_Core.AOSettings) && field.name == "multiBounceInfluence")
                    {
#if !(UNITY_5_1 || UNITY_5_0)
                        if (m_HBAO.generalSettings.integrationStage == HBAO_Core.IntegrationStage.BeforeReflections)
                        {
                            continue;
                        }
#endif
                        if (!m_HBAO.aoSettings.useMultiBounce)
                        {
                            continue;
                        }
                    }
                    // warn about distance falloff greater than max distance
                    else if (group.Key.FieldType == typeof(HBAO_Core.AOSettings) && field.name == "perPixelNormals")
                    {
                        if (m_HBAO.aoSettings.distanceFalloff > m_HBAO.aoSettings.maxDistance)
                        {
                            GUILayout.Space(6.0f);
                            EditorGUILayout.HelpBox("Distance Falloff shoudn't be superior to Max Distance", MessageType.Warning);
                        }
                    }
                    // hide albedoMultiplier when not in deferred
                    else if (group.Key.FieldType == typeof(HBAO_Core.ColorBleedingSettings) && field.name == "albedoMultiplier")
                    {
#if UNITY_5_5_OR_NEWER
                        RenderingPath renderingPath = m_HBAO.GetComponent<Camera>().renderingPath;
                        TierSettings tierSettings = EditorGraphicsSettings.GetTierSettings(EditorUserBuildSettings.selectedBuildTargetGroup, Graphics.activeTier);
                        if (renderingPath != RenderingPath.DeferredShading &&
                            (renderingPath != RenderingPath.UsePlayerSettings ||
                            tierSettings.renderingPath != RenderingPath.DeferredShading))
                        {
                            continue;
                        }
#else
                        RenderingPath renderingPath = m_HBAO.GetComponent<Camera>().renderingPath;
                        if (renderingPath != RenderingPath.DeferredShading &&
                            (renderingPath != RenderingPath.UsePlayerSettings ||
                            PlayerSettings.renderingPath != RenderingPath.DeferredShading))
                        {
                            continue;
                        }
#endif
                    }

                    EditorGUILayout.BeginHorizontal();
                    GUILayout.Space(12.0f);
                    EditorGUILayout.PropertyField(field);
                    EditorGUILayout.EndHorizontal();
                }
            }
        }
        EditorGUILayout.EndVertical();

        serializedObject.ApplyModifiedProperties();
    }

    private void SetStyles()
    {
        // set banner label style
        m_TitleLabelStyle = new GUIStyle(GUI.skin.label);
        m_TitleLabelStyle.alignment = TextAnchor.MiddleCenter;
        m_TitleLabelStyle.contentOffset = new Vector2(0f, 0f);

        // get shuriken module title style
        GUIStyle skurikenModuleTitleStyle = "ShurikenModuleTitle";

        // clone it as to not interfere with the original, and adjust it
        m_SettingsGroupStyle = new GUIStyle(skurikenModuleTitleStyle);
        m_SettingsGroupStyle.font = (new GUIStyle("Label")).font;
        m_SettingsGroupStyle.fontStyle = FontStyle.Bold;
        m_SettingsGroupStyle.border = new RectOffset(15, 7, 4, 4);
        m_SettingsGroupStyle.fixedHeight = 22;
        m_SettingsGroupStyle.contentOffset = new Vector2(10f, -2f);
    }
}
