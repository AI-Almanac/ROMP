"""Side-by-side portrait (heatmap) panels of binned BSS and AUC for each model."""
import os
import numpy as np
import matplotlib.pyplot as plt
from momp.utils.visual import portrait_plot
from momp.io.dict import extract_binned_dict
from momp.utils.printing import tuple_to_str


def panel_portrait_bss_auc(result_binned, verification_window, *, dir_fig, show_panel=True, **kwargs):
    """
    Draw portrait plots of binned BSS (%) and AUC, with a climatology row, for each model.

    The figure is shown if ``show_panel`` is True and always saved to ``dir_fig``.
    Returns ``(fig, (ax1, ax2))``.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # load binned BSS, add climatology on top row
    arr, row_labels, col_labels = extract_binned_dict(result_binned, 'Fair_Brier_Skill_Score')
    data = np.vstack([np.zeros(arr.shape[1]), arr])
    data *= 100
    row_labels = ["Climatology"] + row_labels

    fig, ax1, im = portrait_plot(data, col_labels, row_labels, fig=fig, ax=ax1, 
                                 annotate=True, annotate_data=data, title=r'(a) Brier Skill Score ($\%$)', 
                                 colorbar_off=True)
    ax1.set_xlabel('Forecast window (days)')

    # load binned AUC, add climatology on top row
    arr, row_labels, col_labels = extract_binned_dict(result_binned, 'AUC')
    arr_clim, _, _ = extract_binned_dict(result_binned, 'AUC_ref')
    print("arr = ", arr)
    print("arr_clim = ", arr_clim)
    data = np.vstack([arr_clim[0], arr])
    row_labels = ["Climatology"] + row_labels

    fig, ax2, im = portrait_plot(data, col_labels, row_labels, fig=fig, ax=ax2, 
                                 annotate=True, annotate_data=data, title='(b) AUC', 
                                 colorbar_off=True)
    ax2.set_xlabel('Forecast window (days)')

    fig.tight_layout()
    if show_panel:
        plt.show()

    # save figure
    window_str = tuple_to_str(verification_window)
    figure_filename = f"panel_portrait_BSS_AUC_{window_str}.png"
    figure_filename = os.path.join(dir_fig, figure_filename)
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{figure_filename}'")

    return fig, (ax1, ax2)


if __name__ == "__main__":

    import pandas as pd

    cfg, setting = get_cfg(), get_setting()

    results = {}

    model_list = cfg.model_list
    max_forecast_day = cfg.max_forecast_day
    verification_window = cfg.verification_window_list[0]
    window_str = tuple_to_str(verification_window)

    for model in model_list:
        fout = os.path.join(cfg.dir_out,"binned_skill_scores_{}_{}.csv")
        fout = fout.format(model, window_str)
        df = pd.read_csv(fout)
        dic = df.to_dict(orient='list')
        results[model] = dic

    panel_portrait_bss_auc(results, verification_window, **vars(cfg))
