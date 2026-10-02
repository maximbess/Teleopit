"""Rank-local evidence for the first non-finite rollout value (no collectives)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch


class NonfiniteDiagnostics:
    def __init__(self, env, log_dir, rank):
        self.env = env.unwrapped
        self.directory = Path(log_dir or "outputs") / "nonfinite_diagnostics"
        self.rank = int(rank)
        self.attempted = False

    def context(self):
        """Small pre-step snapshot, including the stage before an automatic reset."""
        command = self.env.command_manager.get_term("ladder")
        return {name: value.detach().clone() for name in
                ("phase", "motion_stage", "reference_time", "stage_elapsed", "attached", "held_rung")
                if isinstance(value := getattr(command, name, None), torch.Tensor)}

    def check(self, values, *, iteration, rollout_step, when, observations, actions=None,
              before=None):
        masks = {name: ~torch.isfinite(value) for name, value in values.items()}
        # One host synchronization per check; no file I/O or CPU state copies on success.
        if not masks or not torch.stack([mask.any() for mask in masks.values()]).any().item():
            return
        message = f"Non-finite rollout value (NaN/Inf), rank={self.rank}, iteration={iteration}, {when}"
        if not self.attempted:
            self.attempted = True
            try:
                path = self._save(values, masks, iteration, rollout_step, when,
                                  observations, actions, before)
                message += f"; diagnostic: {path}"
            except Exception as exc:
                # A diagnostic failure must not replace the original training error.
                message += f"; diagnostic save failed: {type(exc).__name__}: {exc}"
        print(message, flush=True)
        raise ValueError(message)

    def _save(self, values, masks, iteration, rollout_step, when, observations, actions, before):
        bad_ids = sorted({int(i) for mask in masks.values()
                          for i in mask.reshape(mask.shape[0], -1).any(dim=1).nonzero().flatten().cpu().tolist()})
        selected = bad_ids[:16]
        payload = {}
        capture_errors = {}

        def capture(name, getter):
            try:
                # mjlab physics fields can be TorchArray proxies, not Tensor
                # subclasses. detach() exposes their shared torch storage.
                value = getter().detach()
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Expected a tensor or mjlab tensor proxy")
                ids = torch.tensor(selected, device=value.device)
                payload[name] = value.index_select(0, ids).cpu().clone()
            except Exception as exc:
                capture_errors[name] = f"{type(exc).__name__}: {exc}"

        for name, value in values.items():
            capture(name, lambda value=value: value)
        for name, value in observations.items():
            capture(f"observations/{name}", lambda value=value: value)
        if actions is not None:
            capture("actions", lambda: actions)
        for name, value in (before or {}).items():
            capture(f"before/{name}", lambda value=value: value)
        sim = self.env.sim
        for name in ("qpos", "qvel", "qacc", "ctrl", "xfrc_applied", "eq_active"):
            capture(f"physics/{name}", lambda name=name: getattr(sim.data, name))
        capture("episode_length_buf", lambda: self.env.episode_length_buf)
        capture("applied_actions", lambda: self.env.action_manager.action)
        command = self.env.command_manager.get_term("ladder")
        for name in ("phase", "motion_stage", "reference_time", "stage_elapsed", "hold_elapsed",
                     "attached", "held_rung", "motion_failed", "finished", "reference_joint_pos"):
            capture(f"command/{name}", lambda name=name: getattr(command, name))
        disturbances = getattr(command, "disturbances", None)
        if disturbances is not None:
            for name in ("gravity", "gravity_scale", "force", "remaining", "clean", "push_count"):
                capture(f"disturbances/{name}", lambda name=name: getattr(disturbances, name))

        findings = []
        manager = self.env.observation_manager
        for name, mask in masks.items():
            indices = mask.nonzero().cpu()
            if not len(indices):
                continue
            group = name.removeprefix("observations/") if name.startswith("observations/") else None
            components = []
            for index in indices[:64].tolist():
                item = {"index": index, "env_id": index[0],
                        "value": str(values[name][tuple(index)].item())}
                if group is not None:
                    try:
                        axis = self.env.cfg.observations[group].concatenate_dim
                        axis = axis if axis < 0 else axis + 1  # Batch dimension.
                        offset = 0
                        for term, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
                            width = shape[axis if axis < 0 else axis - 1]
                            if offset <= index[axis] < offset + width:
                                item.update(term=term, term_axis_index=index[axis] - offset)
                                break
                            offset += width
                    except (AttributeError, KeyError, IndexError, TypeError):
                        pass  # Raw indices are still exact if layout metadata is unavailable.
                components.append(item)
            findings.append({"field": name, "count": len(indices), "components": components,
                             "components_truncated": len(indices) > 64})

        metadata = {
            "version": 1, "rank": self.rank, "local_rank": os.environ.get("LOCAL_RANK"),
            "pid": os.getpid(), "iteration": iteration, "rollout_step": rollout_step,
            "when": when, "bad_env_ids": bad_ids, "captured_env_ids": selected,
            "capture_limit": 16, "findings": findings, "capture_errors": capture_errors,
            "reference_signature": getattr(command, "reference_signature", None),
            "nominal_gravity": (sim.mj_model.opt.gravity.tolist() if hasattr(sim, "mj_model") else None),
            "motion_stage_names": {"0": "STABILIZE", "1": "PREPARE", "2": "RELEASE",
                                   "3": "TRANSFER", "4": "HOLD", "5": "DONE"},
            "state_timing": "Physics/command state at detection; after_env_step may include automatic resets. before/* precedes that step.",
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"rank{self.rank}_iter{iteration}_step{rollout_step}_{time.time_ns()}"
        path = self.directory / f"{stem}.pt"
        torch.save({"metadata": metadata, "tensors": payload}, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path.resolve()
