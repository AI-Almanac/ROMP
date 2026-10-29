"""Dataclasses holding per-case (Case) and global (Setting) configuration."""
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class Case:
    """One model/verification-window combination, built from the config."""

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

    probabilistic: bool = field(default=False)
    members: Optional[tuple[int, ...]] = field(default=None)

    onset_percentage_threshold: float = field(default=0.5)

    verification_window: tuple[int, int] = field(default=(1,15))
    tolerance_days: int = field(default=3)
    max_forecast_day: int = field(default=30)
    day_bins: Optional[tuple[tuple[int, int], ...]] = field(default=((1, 5), (6, 10), (11, 15)))

    years: Optional[Union[tuple[int, ...], str]] = field(default=None)
    years_clim: Optional[Union[tuple[int, ...], str]] = field(default=None)

    earliest_onset_date: Optional[tuple[int, int]] = field(default=None)

    region: str = field(default=None)

    file_pattern: str = field(default=None)
    unit_cvt: Optional[float] = field(default=None)
    # Canonical name -> name in the model files, e.g. {"init_time": "time"}.
    model_dims: Optional[dict] = field(default=None)

    def update(self, updates):
        """Set the fields named in updates; keys that are not fields are ignored."""
        for key, value in updates.items():
            if key in self.__annotations__:
                setattr(self, key, value)


@dataclass
class Setting:
    """Run-wide settings shared by all cases (paths, dates, metrics and output switches)."""

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

    onset_definition: Optional[str] = field(default=None)
    onset_season: Optional[str] = field(default=None)
    threshold_years: Optional[tuple[int, int]] = field(default=None)
    thresholds: Optional[dict] = field(default=None)
    obs_day_boundary: Optional[str] = field(default=None)
    model_lead_label: Optional[str] = field(default=None)
    check_alignment: bool = field(default=True)
    dry_spell_check: bool = field(default=True)
    forecast_history: Optional[str] = field(default=None)
    threshold_source: Optional[str] = field(default=None)
    onset_backend: str = field(default="numba")
    # Set by the loader in definition mode: the (first, last) leads searched.
    lead_window: Optional[tuple[int, int]] = field(default=None)
    # Set by the loader: True when the config uses the old flat onset keys.
    onset_legacy: bool = field(default=False)

    init_days: tuple = field(default=(0,3))
    date_filter_year: int = field(default=2024)

    MAE: bool = field(default=False)
    FAR: bool = field(default=False)
    MR: bool = field(default=False)
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
        """Set the fields named in updates; keys that are not fields are ignored."""
        for key, value in updates.items():
            if key in self.__annotations__:
                setattr(self, key, value)
