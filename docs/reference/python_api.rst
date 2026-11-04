Python interface
================

The user-facing names, each linking to its entry in the module reference (:doc:`../api`). Only the public interface is
documented: a module's ``__all__`` lists the names that are.

Running
-------

.. autosummary::

   hcwf.run
   hcwf.check
   hcwf.api.run
   hcwf.api.check
   hcwf.api.RunResult

Config objects
--------------

The run config as Python objects: ``Config`` and one class per section. ``Config.to_yaml`` writes the run file and
``Config.to_flat`` gives the validated, resolved flat dict the pipeline reads. The keys of each class are the keys of the
:doc:`config key reference <config_keys>`.

.. autosummary::

   hcwf.config.Config
   hcwf.config.Dataset
   hcwf.config.Mask
   hcwf.config.Reference
   hcwf.config.Onset
   hcwf.config.Units
   hcwf.config.Blend
   hcwf.config.Evaluation
   hcwf.config.Metrics
   hcwf.config.Output
   hcwf.config.UNSET

Blends
------

.. autosummary::

   hcwf.blend.training.train
   hcwf.blend.training.TrainingResult
   hcwf.blend.trained.TrainedBlend
   hcwf.api.forecast

Errors
------

Every error HCWF raises on purpose is defined in `hcwf.errors`.

.. autosummary::

   hcwf.errors.HCWFConfigError
   hcwf.errors.ConfigSchemaError
   hcwf.errors.OnsetConfigError
   hcwf.errors.DatasetError
   hcwf.errors.RunExistsError
   hcwf.errors.BlendFileError
   hcwf.errors.IncompatibleForecast
