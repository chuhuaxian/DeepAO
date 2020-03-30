using UnityEngine;

public class HBAO_MinMaxSliderAttribute : PropertyAttribute
{
    public readonly float max;
    public readonly float min;

    public HBAO_MinMaxSliderAttribute(float min, float max)
    {
        this.min = min;
        this.max = max;
    }
}
