using UnityEngine;

public class HBAOControl : MonoBehaviour {

    public HBAO hbao;
    public UnityEngine.UI.Slider aoRadiusSlider;

    public void ToggleShowAO () {
        if (hbao.generalSettings.displayMode != HBAO.DisplayMode.Normal) {
            HBAO.GeneralSettings settings = hbao.generalSettings;
            settings.displayMode = HBAO.DisplayMode.Normal;
            hbao.generalSettings = settings;
        } else {
            HBAO.GeneralSettings settings = hbao.generalSettings;
            settings.displayMode = HBAO.DisplayMode.AOOnly;
            hbao.generalSettings = settings;
        }
    }

    public void UpdateAoRadius () {
        HBAO.AOSettings settings = hbao.aoSettings;
        settings.radius = aoRadiusSlider.value;
        hbao.aoSettings = settings;
    }
}
