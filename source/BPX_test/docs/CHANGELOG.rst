Changelog
---------

0.2.0 (2026-09-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Split the shared BPX robot and policy interface from locomotion, standing, and Init task configurations.
* Added ``BPX-Locomotion-v0``, ``BPX-Stand-v0``, and ``BPX-Init-v0`` while retaining ``BPX-Test-v0``.
* Added task-specific RSL-RL experiment directories and made the ``--experiment_name`` override effective.
* Replaced Init's fixed two-second motion target with state-driven height, joint-posture, four-foot support, and post-recovery stability rewards.
* Added body-frame front-foot width, gated hip-roll posture, and gated foot-slide terms to prevent Init from learning a narrow front support stance.

0.1.0 (2026-09-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
* Replaced the cart-pole template task with the BPX quadruped flat-terrain velocity-tracking locomotion task (``BPX-Test-v0``)
