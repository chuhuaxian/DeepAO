using UnityEditor;

[CustomEditor(typeof(HBAO_Integrated))]
public class HBAO_Integrated_Editor : HBAO_Core_Editor
{
    public override void OnInspectorGUI()
    {
        DrawGUI(true);
    }
}
