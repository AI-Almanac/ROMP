"""Validation of the ROMP configuration dictionary (consistency and logic checks)."""
import os
from datetime import datetime
from types import SimpleNamespace


# dry_extent > dry_spell

# model name no space

#window should include day_bins

# probabilistic -- must be only one window in window_list, window end == max_forecast_day

#Search window (days) after onset to check for the dry spell. Must be >= dry_spell.
# check trailing comma in _list


# plot_bar_bss_rpss_auc = True, <- save_csv_score = True

# plot_panel_heatmap_error = True <--- veri window list must have >1 bins

# 'plot_panel_heatmap_skill' <---  veri window list must be single item
# 'plot_bar_bss_rpss_auc' <---  veri window list must be single item


class ROMPConfigError(Exception):
    """Custom exception for configuration failures."""
    pass

class ROMPValidator:
    """Collects consistency errors for a configuration dict and raises them together."""

    def __init__(self, config_dict):
        """Wrap the config dict in a namespace and start with no errors."""
        # Convert the dictionary to an object so we can use self.cfg.param
        self.cfg = SimpleNamespace(**config_dict)
        self.errors = []

    def _add_error(self, msg):
        """Record one error message."""
        self.errors.append(msg)

    def validate(self):
        """Runs all consistency and logic checks."""
        self._check_temporal()
        self._check_list_lengths()
        self._check_model_names()
        self._check_tuple_ranges()
        self._check_forecast_limits()
        self._check_probabilistic_constraints()
        self._check_monsoon_logic()

        if self.errors:
            report = "\n".join([f"  • {err}" for err in self.errors])
            raise ROMPConfigError(f"ROMP Configuration is invalid:\n{report}")
        return True

    def _check_temporal(self):
        """Check start_date/end_date ordering and the climatology year range."""
        try:
            sd = datetime(*self.cfg.start_date)
            ed = datetime(*self.cfg.end_date)
            if sd >= ed:
                self._add_error(f"start_date ({sd.date()}) is after end_date ({ed.date()})")

            sy = self.cfg.start_year_clim
            ey = self.cfg.end_year_clim
            if sd > ed:
                self._add_error(f"start_year_clim ({sy}) is after end_year_clim ({ey})")

        except Exception as e:
            self._add_error(f"Date formatting error: {e}")

    def _check_list_lengths(self):
        """Check that per-model _list entries match model_list (tolerance_days_list: verification windows)."""
        # We use the length of model_list as the 'Master' length
        master_len = len(self.cfg.model_list)

        # Access the underlying dictionary to find all keys ending in '_list'
        for key, value in vars(self.cfg).items():
            if key.endswith('_list') and isinstance(value, (list, tuple)):
                if key not in ('verification_window_list', 'tolerance_days_list'):
                    if len(value) != master_len:
                        self._add_error(f"Length Mismatch: '{key}' has {len(value)} items, \
                                        expected {master_len}")
                if key in ('tolerance_days_list',):
                    if len(value) != len(self.cfg.verification_window_list):
                        self._add_error(f"Length Mismatch: '{key}' has {len(value)} items, \
                                        expected {len(self.cfg.verification_window_list)}")

    def _check_model_names(self):
        """Check that model names contain no spaces."""
        # 1. Ensure no spaces in model names (critical for file paths/naming)
        for model in self.cfg.model_list:
            if " " in model:
                self._add_error(f"Invalid model name: '{model}' (Spaces are not allowed).")

    def _check_tuple_ranges(self):
        """Check that each (start, end) pair in verification windows and day_bins has start < end."""
        # 2. Ensure (start, end) pairs are logically ordered (start < end)
        for attr_name in ['verification_window_list', 'day_bins']:
            windows = getattr(self.cfg, attr_name)
            for window in windows:
                if window[0] >= window[1]:
                    self._add_error(f"Invalid range in {attr_name}: {window}. \
                            First number must be smaller than the second.")

    def _check_forecast_limits(self):
        """Check that no verification window extends beyond max_forecast_day."""
        # 3. Largest window value cannot exceed max_forecast_day
        max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
        if max_v_day > self.cfg.max_forecast_day:
            self._add_error(f"Verification window limit ({max_v_day}) \
                    exceeds max_forecast_day ({self.cfg.max_forecast_day}).")

    def _check_probabilistic_constraints(self):
        """In probabilistic mode, check the largest window end covers the largest day_bin end."""
        # Using .get() in case the key is missing to avoid AttributeError
        is_prob = getattr(self.cfg, 'probabilistic', False)

        if is_prob:
            # 5. Window max cannot be smaller than day_bins max
            max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
            max_bin_day = max([pair[1] for pair in self.cfg.day_bins])
            if max_v_day < max_bin_day:
                self._add_error(f"Window mismatch: Largest verification day \
                        ({max_v_day}) is smaller than largest day_bin ({max_bin_day}).")

    def _check_monsoon_logic(self):
        """Check dry_extent against dry_spell and that thresh_file exists if given."""
        if self.cfg.dry_extent > 0 and self.cfg.dry_extent < self.cfg.dry_spell:
            self._add_error(f"Logic Conflict: \
                    dry_extent ({self.cfg.dry_extent}) < dry_spell ({self.cfg.dry_spell})")

        if self.cfg.thresh_file and not os.path.exists(self.cfg.thresh_file):
            self._add_error(f"File missing: thresh_file not found at '{self.cfg.thresh_file}'")

    def _check_graphics_dependencies(self):
        """Validates that plot settings match data availability and output requirements."""

        # 1. Bar plot requires CSV data for input
        if getattr(self.cfg, 'plot_bar_bss_rpss_auc', False):
            if not getattr(self.cfg, 'save_csv_score', False):
                self._add_error("Graphics Conflict: 'plot_bar_bss_rpss_auc' \
                        requires 'save_csv_score' to be True.")

        # 2. Panel Error Heatmap requires multiple items to compare (Matrix layout)
        if getattr(self.cfg, 'plot_panel_heatmap_error', False):
            if len(self.cfg.verification_window_list) <= 1 or len(self.cfg.model_list) <= 1:
                self._add_error("Graphics Conflict: 'plot_panel_heatmap_error' \
                        requires >1 model AND >1 verification window.")

        # 3. Panel Skill Heatmap logic
        if getattr(self.cfg, 'plot_panel_heatmap_skill', False):
            if len(self.cfg.verification_window_list) != 1:
                self._add_error("Graphics Conflict: 'plot_panel_heatmap_skill' \
                        only supports exactly ONE verification window.")

        # 4. Bar plots for Skill Scores (BSS/RPSS/AUC)
        if getattr(self.cfg, 'plot_bar_bss_rpss_auc', False):
            if len(self.cfg.verification_window_list) != 1:
                self._add_error("Graphics Conflict: 'plot_bar_bss_rpss_auc' \
                        only supports exactly ONE verification window.")


if __name__ == "__main__":
    from momp.lib.loader import get_cfg
    cfg = get_cfg()
    try:
        validator = ROMPValidator(vars(cfg))
        validator.validate()
        print("Configuration validated!")
    except ROMPConfigError as e:
        print(e)
