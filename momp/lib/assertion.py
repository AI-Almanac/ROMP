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
    def __init__(self, config_dict):
        # Convert the dictionary to an object so we can use self.cfg.param
        self.cfg = SimpleNamespace(**config_dict)
        self.errors = []

    def _add_error(self, msg):
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
#        self._check_graphics_dependencies()
        
        if self.errors:
            report = "\n".join([f"  • {err}" for err in self.errors])
            raise ROMPConfigError(f"ROMP Configuration is invalid:\n{report}")
        return True

    def _check_temporal(self):
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
        # 1. Ensure no spaces in model names (critical for file paths/naming)
        for model in self.cfg.model_list:
            if " " in model:
                self._add_error(f"Invalid model name: '{model}' (Spaces are not allowed).")

    def _check_tuple_ranges(self):
        # 2. Ensure (start, end) pairs are logically ordered (start < end)
        for attr_name in ['verification_window_list', 'day_bins']:
            windows = getattr(self.cfg, attr_name)
            for window in windows:
                if window[0] >= window[1]:
                    self._add_error(f"Invalid range in {attr_name}: {window}. \
                            First number must be smaller than the second.")

    def _check_forecast_limits(self):
        # 3. Largest window value cannot exceed max_forecast_day
        max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
        if max_v_day > self.cfg.max_forecast_day:
            self._add_error(f"Verification window limit ({max_v_day}) \
                    exceeds max_forecast_day ({self.cfg.max_forecast_day}).")

    def _check_probabilistic_constraints(self):
        # Using .get() in case the key is missing to avoid AttributeError
        is_prob = getattr(self.cfg, 'probabilistic', False)

        if is_prob:
            ## 4. Probabilistic runs must have exactly one verification window
            #if len(self.cfg.verification_window_list) != 1:
            #    self._add_error("Probabilistic Mode Error: \
            #            verification_window_list must contain exactly one tuple.")

            # 5. Window max cannot be smaller than day_bins max
            max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
            max_bin_day = max([pair[1] for pair in self.cfg.day_bins])
            if max_v_day < max_bin_day:
                self._add_error(f"Window mismatch: Largest verification day \
                        ({max_v_day}) is smaller than largest day_bin ({max_bin_day}).")

    def _check_monsoon_logic(self):
        # Resolving the rule runs its bound and cross-field validation (for
        # "legacy" this includes the dry_extent >= dry_spell constraint).
        from momp.stats.onset_rule import resolve_rule

        try:
            rule = resolve_rule(
                getattr(self.cfg, "onset_rule", None),
                getattr(self.cfg, "onset_rule_params", None),
                wet_init=getattr(self.cfg, "wet_init", None),
                wet_spell=getattr(self.cfg, "wet_spell", None),
                dry_spell=getattr(self.cfg, "dry_spell", None),
                dry_extent=getattr(self.cfg, "dry_extent", None),
            )
        except (ValueError, TypeError) as e:
            self._add_error(f"Invalid onset rule: {e}")
        else:
            max_day = getattr(self.cfg, "max_forecast_day", None)
            if max_day is not None and rule.lookahead >= max_day:
                self._add_error(
                    f"Onset rule '{rule.label()}' needs {rule.lookahead} days of lookahead, which is "
                    f">= max_forecast_day ({max_day}); forecasts must supply at least "
                    f"{max_day + rule.lookahead} steps and few candidate days will be verifiable."
                )

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
