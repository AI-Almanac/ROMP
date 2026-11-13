""" preprocessing, formatting and standardizing input data"""
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Union

import numpy as np
#import xcdat as xc
#from ARMP.params.region_def import domain


@dataclass
class Case:
    case_name: str = field(default=None)

    model: str = field(default=None)
    model_var: str = field(default=None)
    model_dir: str = field(default="data")

    obs: str = field(default=None)
    obs_var: str = field(default=None)

    ref_model: Optional[str] = field(default=None)
    ref_model_var: Optional[str] = field(default=None)

    thresh_file: Optional[str] = field(default=None)
    thresh_var: Optional[str] = field(default=None)

    wet_init: float = field(default=1.0)
    wet_threshold: float = field(default=20.0)
    wet_spell: int = field(default=3)
    dry_threshold: float = field(default=1.0)
    dry_spell: int = field(default=0)
    dry_extent: int = field(default=0)

    # Onset definition: a registered momp.stats.onset_rule name plus its
    # parameters. "legacy" (the default) reads wet_init/wet_spell/dry_spell/
    # dry_extent above, so existing configs are unchanged.
    onset_rule: str = field(default="legacy")
    onset_rule_params: Optional[dict[str, Any]] = field(default=None)

    probabilistic: bool = field(default=False)
    members: Optional[tuple[int, ...]] = field(default=None)

    onset_percentage_threshold: float = field(default=0.5)
    
    verification_window: tuple[int, int] = field(default=(1,15))
    tolerance_days: int = field(default=3)
    max_forecast_day: int = field(default=30)
    day_bins: Optional[tuple[tuple[int, int], ...]] = field(default=((1, 5), (6, 10), (11, 15)))

    years: Optional[Union[tuple[int, ...], str]] = field(default=None)
    years_clim: Optional[Union[tuple[int, ...], str]] = field(default=None)

    mok: Optional[tuple[int, int]] = field(default="")

    region: str = field(default=None)

    file_pattern: str = field(default=None)
    unit_cvt: Optional[float] = field(default=None)
    # Canonical name -> name in the model files, e.g. {"init_time": "time"}.
    model_dims: Optional[dict] = field(default=None)


    def update(self, updates):
        for key, value in updates.items():
            if key in self.__annotations__:
                setattr(self, key, value)


#    def __post_init__(self):
#        self.fn_var = self.tag_var
#        self.fn_var_out = self.tag_var_out
#        self.fn_freq = self.tag_freq
#        self.fn_list = self.tag_list



@dataclass
class Setting:
    work_dir: str = field(default="~/")
    pkg_dir: str = field(default="~/")

    layout: list = field(default_factory=list)
    model_list: tuple[str, ...] = field(default=None)
    verification_window_list: tuple[tuple[int, int], ...] = field(default=None)

    start_date: tuple[int, int, int] = field(default=(2019,5,1))
    end_date: tuple[int, int, int] = field(default=(2024,10,31))
    start_year_clim: int = field(default=1980)
    end_year_clim: int = field(default=2000)
    fallback_date: Optional[tuple[int, int]] = field(default=None)

    init_days: tuple = field(default=(0,3))
    date_filter_year: int = field(default=2024)

    MAE: bool = field(default=False)
    FAR: bool = field(default=False)
    MR: bool = field(default=False)
    #CSI: bool = field(default=False)

    #probabilistic: bool = field(default=False)
    #members: Optional[tuple[int, ...]] = field(default=None)

    BS: bool = field(default=False)
    RPS: bool = field(default=False)
    AUC: bool = field(default=False)

    skill_score: bool = field(default=False)

    obs_dir: str = field(default="../data")
    ref_model_dir: Optional[str] = field(default="../data")

    shpfile_dir: Optional[str] = field(default=None)
    nc_mask: Optional[str] = field(default=None)

    obs_file_pattern: str = field(default="{}.nc")
    ref_model_file_pattern: Optional[str] = field(default="{}.nc")
    
    obs_unit_cvt: Optional[float] = field(default=None)
    ref_model_unit_cvt: Optional[float] = field(default=None)

    #dir_in: str = field(default="data")
    dir_out: str = field(default="../output")
    dir_fig: str = field(default="../figures")

    save_fig: bool = field(default=False)

    save_nc_spatial_far_mr_mae: bool = field(default=False)
    save_csv_score: bool = field(default=False)
    save_nc_climatology: bool = field(default=False)

    polygon: bool = field(default=False)

    plot_spatial_far_mr_mae: bool = field(default=False)
    plot_heatmap_bss_auc: bool = field(default=False)
    plot_reliability: bool = field(default=False)
    plot_portrait: bool = field(default=False)
    plot_climatology_onset: bool = field(default=False)
    plot_panel_heatmap_error: bool = field(default=False)
    plot_panel_heatmap_skill: bool = field(default=False)
    plot_bar_bss_rpss_auc: bool = field(default=False)
    show_plot: bool = field(default=False)
    show_panel: bool = field(default=False)

    parallel: bool = field(default=False)

    debug: bool = field(default=False)
    project_name: str = field(default="ROMP application project")

    def update(self, updates):
        for key, value in updates.items():
            if key in self.__annotations__:
                setattr(self, key, value)


