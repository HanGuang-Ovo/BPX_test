"""Persist custom terrain curricula alongside the standard PPO checkpoint."""

from rsl_rl.runners import OnPolicyRunner
import torch

from tensorboard_metrics import ScalarFilterWriter


class CurriculumOnPolicyRunner(OnPolicyRunner):
    def log(self, locs, width=80, pad=35):
        pattern = self.cfg.get("tensorboard_scalar_filter")
        if not pattern or self.logger_type != "tensorboard":
            return super().log(locs, width=width, pad=pad)
        # The upstream runner creates its writer inside learn(). Wrap only its
        # logging call so writer flush/close/save operations retain their API.
        writer = self.writer
        self.writer = ScalarFilterWriter(writer, pattern)
        try:
            return super().log(locs, width=width, pad=pad)
        finally:
            self.writer = writer

    def _curricula(self):
        manager = getattr(self.env.unwrapped, "curriculum_manager", None)
        if manager is None:
            return {}
        return {name: cfg.func for name, cfg in zip(manager._term_names, manager._term_cfgs)
                if hasattr(cfg.func, "state_dict") and hasattr(cfg.func, "load_state_dict")}

    def save(self, path, infos=None):
        curricula = self._curricula()
        if curricula:
            infos = dict(infos or {})
            infos["bpx_curricula"] = {name: term.state_dict() for name, term in curricula.items()}
        super().save(path, infos)

    def load(self, path, load_optimizer=True):
        infos = super().load(path, load_optimizer=load_optimizer)
        # Fine-tuning transfers weights, not the old task's terrain or streaks.
        if load_optimizer and infos and "bpx_curricula" in infos:
            curricula = self._curricula()
            for name, state in infos["bpx_curricula"].items():
                if name in curricula:
                    curricula[name].load_state_dict(state)
            # Install restored levels before new physics is collected. Do not
            # count the freshly initialized environment as a completed episode.
            with torch.inference_mode():
                self.env.unwrapped.episode_length_buf.zero_()
                self.env.reset()
        return infos
