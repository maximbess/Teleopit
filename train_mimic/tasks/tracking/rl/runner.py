import os
import pathlib
import statistics
import time
from itertools import chain
from math import exp, log

import torch
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan


def _one_based_iteration_range(start_iteration: int, total_iterations: int) -> range:
    """Return the inclusive 1-based iteration range up to the target total."""
    if total_iterations < start_iteration:
        raise ValueError(
            "num_learning_iterations must be >= the completed iteration count when resuming. "
            f"Got total_iterations={total_iterations}, start_iteration={start_iteration}."
        )
    return range(start_iteration + 1, total_iterations + 1)


def _resolve_total_iterations(
    start_iteration: int, num_learning_iterations: int
) -> int:
    """Return the cumulative 1-based target iteration after running more iterations."""
    if num_learning_iterations < 0:
        raise ValueError(
            "num_learning_iterations must be non-negative. "
            f"Got num_learning_iterations={num_learning_iterations}."
        )
    return start_iteration + num_learning_iterations


def _format_duration(seconds: float) -> str:
    """Format elapsed/remaining seconds without wrapping after 24 hours."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ordered_episode_extra_keys(ep_extras: list[dict]) -> tuple[str, ...]:
    """Return every logged episode key in first-seen order.

    MJLab emits an empty ``extras["log"]`` dictionary on policy steps without
    resets.  A reset can occur later in the same rollout, so inspecting only
    the first dictionary silently drops those episode metrics.
    """

    return tuple(
        dict.fromkeys(key for episode_info in ep_extras for key in episode_info)
    )


def _distribution_parameter_and_bounds(
    distribution,
) -> tuple[torch.nn.Parameter, tuple[float, float]]:
    """Return the learnable Gaussian scale parameter and its native bounds."""

    std_type = getattr(distribution, "std_type", None)
    if std_type == "scalar":
        return distribution.std_param, tuple(float(x) for x in distribution.std_range)
    if std_type == "log":
        return distribution.log_std_param, tuple(
            float(x) for x in distribution.log_std_range
        )
    raise TypeError(
        "Ladder PPO requires a Gaussian distribution with scalar or log std; "
        f"got std_type={std_type!r}."
    )


def _project_distribution_std(
    distribution,
) -> tuple[torch.nn.Parameter, bool]:
    """Project the raw std parameter so clamp cannot leave it gradient-dead."""

    parameter, (minimum, maximum) = _distribution_parameter_and_bounds(distribution)
    projected = bool(
        torch.any((parameter < minimum) | (parameter > maximum)).item()
    )
    with torch.no_grad():
        parameter.clamp_(min=minimum, max=maximum)
    return parameter, projected


def _raw_distribution_std_mean(distribution) -> torch.Tensor:
    """Return the unclamped learnable std in effective scalar space."""

    parameter, _ = _distribution_parameter_and_bounds(distribution)
    raw_std = torch.exp(parameter) if distribution.std_type == "log" else parameter
    return raw_std.mean()


def _set_distribution_std(distribution, target_std: float) -> torch.nn.Parameter:
    """Set every action dimension to one valid effective standard deviation."""

    if target_std <= 0.0:
        raise ValueError(f"target_std must be positive, got {target_std}")
    parameter, (minimum, maximum) = _distribution_parameter_and_bounds(distribution)
    native_target = log(target_std) if distribution.std_type == "log" else target_std
    if not minimum <= native_target <= maximum:
        effective_bounds = (
            (exp(minimum), exp(maximum))
            if distribution.std_type == "log"
            else (minimum, maximum)
        )
        raise ValueError(
            f"target_std={target_std} is outside configured std range "
            f"{effective_bounds}"
        )
    with torch.no_grad():
        parameter.fill_(native_target)
    return parameter


class LadderOnPolicyRunner(MjlabOnPolicyRunner):
    """Plain PPO runner with persistent, multi-GPU ladder curriculum state."""

    env: RslRlVecEnvWrapper

    def _ladder_command(self):
        return self.env.unwrapped.command_manager.get_term("ladder")

    def _actor_distribution(self):
        distribution = self.alg.get_policy().distribution
        if distribution is None:
            raise RuntimeError("Ladder PPO actor must define a Gaussian distribution")
        return distribution

    def _restore_exploration_after_promotion(self) -> tuple[float, float]:
        """Reset std, its optimizer momentum, and LR for a newly opened phase."""

        target_std = float(self.cfg["actor"]["distribution_cfg"]["init_std"])
        distribution = self._actor_distribution()
        std_parameter = _set_distribution_std(distribution, target_std)
        self.alg.optimizer.state.pop(std_parameter, None)

        target_learning_rate = float(self.cfg["algorithm"]["learning_rate"])
        self.alg.learning_rate = target_learning_rate
        for param_group in self.alg.optimizer.param_groups:
            param_group["lr"] = target_learning_rate
        return target_std, target_learning_rate

    def _project_actor_std(self) -> None:
        """Project actor std and discard optimizer momentum that crossed a bound."""

        std_parameter, projected = _project_distribution_std(
            self._actor_distribution()
        )
        if projected:
            self.alg.optimizer.state.pop(std_parameter, None)

    def _synchronize_ladder_curriculum(self, iteration: int) -> None:
        command = self._ladder_command()
        local_outcomes = command.drain_curriculum_outcomes()
        if self.is_distributed:
            gathered: list[list[int] | None] = [None] * self.gpu_world_size
            torch.distributed.all_gather_object(gathered, local_outcomes)
            outcomes = list(chain.from_iterable(batch or [] for batch in gathered))
        else:
            outcomes = local_outcomes

        promoted = command.update_curriculum(outcomes)
        promotion_std = None
        promotion_learning_rate = None
        if promoted:
            promotion_std, promotion_learning_rate = (
                self._restore_exploration_after_promotion()
            )
        if promoted and self.gpu_global_rank == 0:
            phase_name = command.phase_name(command.max_unlocked_phase)
            print(
                "[INFO] Ladder curriculum promoted to "
                f"phase {command.max_unlocked_phase} ({phase_name}) at "
                f"iteration {iteration}; actor std reset to {promotion_std:.3f}, "
                f"learning rate reset to {promotion_learning_rate:.3g}."
            )

        writer = self.logger.writer
        if writer is not None:
            writer.add_scalar(
                "Curriculum/ladder_unlocked_phase",
                command.max_unlocked_phase,
                iteration,
            )
            writer.add_scalar(
                "Curriculum/ladder_success_rate_100",
                command.curriculum_success_rate,
                iteration,
            )
            writer.add_scalar(
                "Curriculum/ladder_window_fill",
                command.curriculum_window_fill,
                iteration,
            )
            writer.add_scalar(
                "Curriculum/ladder_phase_steps",
                command.curriculum_phase_steps,
                iteration,
            )
            if promoted:
                writer.add_scalar(
                    "Curriculum/ladder_promotion_actor_std",
                    promotion_std,
                    iteration,
                )
                writer.add_scalar(
                    "Curriculum/ladder_promotion_learning_rate",
                    promotion_learning_rate,
                    iteration,
                )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run PPO and evaluate curriculum promotion once per rollout."""

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = _resolve_total_iterations(start_it, num_learning_iterations)
        for it in _one_based_iteration_range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    self.logger.process_env_step(
                        rewards,
                        dones,
                        extras,
                        intrinsic_rewards,
                    )

                self._synchronize_ladder_curriculum(it)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            self._project_actor_std()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            MotionTrackingOnPolicyRunner._log_one_based_iteration(
                self,
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=(
                    self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None
                ),
            )
            if self.logger.writer is not None:
                self.logger.writer.add_scalar(
                    "Policy/raw_mean_std",
                    _raw_distribution_std_mean(self._actor_distribution()),
                    it,
                )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            self.logger.stop_logging_writer()

    def save(self, path: str, infos=None) -> None:
        curriculum_state = self._ladder_command().curriculum_state_dict()
        infos = {**(infos or {}), "ladder_curriculum_state": curriculum_state}
        super().save(path, infos=infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        command = self._ladder_command()
        state = (infos or {}).get("ladder_curriculum_state")
        if state is None:
            if command.cfg.curriculum_enabled:
                raise RuntimeError(
                    "Checkpoint does not contain adaptive ladder curriculum state. "
                    "Start a fresh run instead of resuming a fixed-schedule checkpoint."
                )
            self._project_actor_std()
            return infos
        command.load_curriculum_state_dict(state)
        self._project_actor_std()
        return infos


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        """Run the learning loop using 1-based iteration numbering."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = _resolve_total_iterations(start_it, num_learning_iterations)
        for it in _one_based_iteration_range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )
                    self.logger.process_env_step(
                        rewards, dones, extras, intrinsic_rewards
                    )

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self._log_one_based_iteration(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight
                if self.cfg["algorithm"]["rnd_cfg"]
                else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    def _log_one_based_iteration(
        self,
        *,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        logger = self.logger
        if logger.writer is None:
            return

        collection_size = (
            logger.cfg["num_steps_per_env"] * logger.num_envs * logger.gpu_world_size
        )
        iteration_time = collect_time + learn_time
        logger.tot_timesteps += collection_size
        logger.tot_time += iteration_time

        extras_string = ""
        if logger.ep_extras:
            for key in _ordered_episode_extra_keys(logger.ep_extras):
                infotensor = torch.tensor([], device=logger.device)
                for ep_info in logger.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(logger.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    logger.writer.add_scalar(key, value, it)  # type: ignore[arg-type]
                    extras_string += f"""{f"{key}:":>{pad}} {value:.4f}
"""
                else:
                    logger.writer.add_scalar("Episode/" + key, value, it)  # type: ignore[arg-type]
                    extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}
"""

        for key, value in loss_dict.items():
            logger.writer.add_scalar(f"Loss/{key}", value, it)
        logger.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        logger.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        logger.writer.add_scalar("Perf/total_fps", fps, it)
        logger.writer.add_scalar("Perf/collection_time", collect_time, it)
        logger.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(logger.rewbuffer) > 0:
            if logger.cfg["algorithm"]["rnd_cfg"]:
                logger.writer.add_scalar(
                    "Rnd/mean_extrinsic_reward", statistics.mean(logger.erewbuffer), it
                )
                logger.writer.add_scalar(
                    "Rnd/mean_intrinsic_reward", statistics.mean(logger.irewbuffer), it
                )
                logger.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore[arg-type]
            logger.writer.add_scalar(
                "Train/mean_reward", statistics.mean(logger.rewbuffer), it
            )
            logger.writer.add_scalar(
                "Train/mean_episode_length", statistics.mean(logger.lenbuffer), it
            )
            if logger.logger_type != "wandb":
                logger.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(logger.rewbuffer),
                    int(logger.tot_time),
                )
                logger.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(logger.lenbuffer),
                    int(logger.tot_time),
                )

        log_string = f"""{"#" * width}
"""
        heading = f" Learning iteration {it}/{total_it} "
        log_string += f"\033[1m{heading.center(width)}\033[0m\n\n"

        run_name = logger.cfg.get("run_name")
        log_string += (
            f"""{"Run name:":>{pad}} {run_name}
"""
            if run_name
            else ""
        )
        log_string += (
            f"""{"Total steps:":>{pad}} {logger.tot_timesteps}
"""
            f"""{"Steps per second:":>{pad}} {fps:.0f}
"""
            f"""{"Collection time:":>{pad}} {collect_time:.3f}s
"""
            f"""{"Learning time:":>{pad}} {learn_time:.3f}s
"""
        )

        for key, value in loss_dict.items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}
"""

        if len(logger.rewbuffer) > 0:
            if logger.cfg["algorithm"]["rnd_cfg"]:
                log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(logger.erewbuffer):.2f}
"""
                log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(logger.irewbuffer):.2f}
"""
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(logger.rewbuffer):.2f}
"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(logger.lenbuffer):.2f}
"""

        log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}
"""
        if not print_minimal:
            log_string += extras_string

        done_it = it - start_it
        remaining_it = total_it - it
        eta = logger.tot_time / done_it * remaining_it if done_it > 0 else 0.0
        log_string += (
            f"""{"-" * width}
"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s
"""
            f"""{"Time elapsed:":>{pad}} {_format_duration(logger.tot_time)}
"""
            f"""{"ETA:":>{pad}} {_format_duration(eta)}
"""
        )
        print(log_string)

        if logger.logger_type == "wandb":
            for video in pathlib.Path(logger.log_dir).rglob("*.mp4"):  # type: ignore[arg-type]
                logger.writer.save_video(video, it)  # type: ignore[arg-type]

        logger.ep_extras.clear()

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, filename)
        model = self.alg.get_policy().as_onnx(verbose=False)
        model.to("cpu")
        model.eval()
        dummy_inputs = model.get_dummy_inputs()
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,
            output_names=model.output_names,
            dynamic_axes={},
            dynamo=False,
        )
