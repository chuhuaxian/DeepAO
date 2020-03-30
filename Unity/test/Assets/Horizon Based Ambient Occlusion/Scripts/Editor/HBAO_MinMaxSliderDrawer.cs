using UnityEngine;
using UnityEditor;

[CustomPropertyDrawer(typeof(HBAO_MinMaxSliderAttribute))]
class HBAO_MinMaxSliderDrawer : PropertyDrawer
{

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {

        if (property.propertyType == SerializedPropertyType.Vector2)
        {
            Vector2 range = property.vector2Value;
            float min = range.x;
            float max = range.y;
            HBAO_MinMaxSliderAttribute attr = attribute as HBAO_MinMaxSliderAttribute;
            EditorGUI.BeginChangeCheck();
#if UNITY_5_5_OR_NEWER
            EditorGUI.MinMaxSlider(position, label, ref min, ref max, attr.min, attr.max);
#else
            EditorGUI.MinMaxSlider(label, position, ref min, ref max, attr.min, attr.max);
#endif
            if (EditorGUI.EndChangeCheck())
            {
                range.x = min;
                range.y = max;
                property.vector2Value = range;
            }
        }
        else
        {
            EditorGUI.LabelField(position, label, "Use only with Vector2");
        }
    }
}
