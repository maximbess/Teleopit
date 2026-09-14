# PROJECT_CONTEXT — Teleopit

> Снимок контекста: 2026-09-01. Этот файл предназначен для переноса репозитория в новый ChatGPT Project и быстрого продолжения работы без потери принятых решений.

## 1. Как пользоваться этим файлом

- Сначала прочитать `AGENTS.md`: это наиболее подробный и актуальный набор проектных правил.
- Затем использовать этот файл как краткий handoff по архитектуре, текущему направлению работы, командам и состоянию рабочей копии.
- При расхождении документации и кода источником истины является текущий код, но расхождение нужно исправить сразу во всех связанных местах.
- Не откатывать и не перезаписывать существующие незакоммиченные изменения: в ветке находится большой рабочий блок по RL-обучению лазанию по лестнице.
- Не коммитить изменения автоматически. Автор коммита — текущий настроенный Git user.

## 2. Состояние репозитория на момент передачи

- Локальный путь: `D:\Westlake Internship\teleopit`.
- Активная ветка: `climbing`, отслеживает `origin/climbing`.
- HEAD: `ab843795dc2d8ca67d6de37833517ca467985261` (`init climbing version`, 2026-08-06).
- Версия Python-пакета в `pyproject.toml`: `0.4.0`.
- `origin`: `https://github.com/maximbess/Teleopit.git`.
- `upstream`: `https://github.com/BotRunner64/Teleopit.git`.
- Рабочее дерево уже было грязным до создания этого файла: 16 изменённых файлов и 2 новых файла, в основном вокруг ladder task. `PROJECT_CONTEXT.md` добавлен поверх этих изменений.

Изменённые файлы текущего рабочего блока:

- `AGENTS.md`
- `README.md`
- `docs/docs/reference/architecture.md`
- `docs/docs/tutorials/training.md`
- `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/architecture.md`
- `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/training.md`
- `tests/test_ladder_training.py`
- `tests/test_record_ladder_video.py`
- `tests/test_runner_iteration_numbering.py`
- `train_mimic/scripts/play.py`
- `train_mimic/scripts/record_ladder_video.py`
- `train_mimic/scripts/train_ladder.py`
- `train_mimic/tasks/tracking/config/env.py`
- `train_mimic/tasks/tracking/config/rl.py`
- `train_mimic/tasks/tracking/mdp/ladder.py`
- `train_mimic/tasks/tracking/rl/runner.py`

Новые незакоммиченные файлы:

- `train_mimic/ladder_playback.py`
- `tests/test_play_script.py`

Главное: этот рабочий блок нельзя воспринимать как случайный diff или удалять ради «чистой» ветки. Он содержит новую механику ladder curriculum, soft hand release, новые наблюдения/награды и поддержку ladder playback.

## 3. Назначение проекта

Teleopit — лёгкий, расширяемый, самодостаточный фреймворк whole-body teleoperation для гуманоидных роботов, прежде всего Unitree G1.

Основной runtime-поток:

```text
InputProvider (BVH / Pico 4 VR)
    -> Retargeter (GMR)
    -> ObservationBuilder (167D velcmd_history)
    -> Controller (dual-input TemporalCNN ONNX)
    -> Robot (MuJoCo + PD или Unitree SDK)
```

Отдельно существует пакет `train_mimic` с двумя задачами:

1. `General-Tracking-G1` — imitation/motion tracking по датасету движений.
2. `G1-Ladder-Climb-RL` — чистое PPO-обучение лазанию по лестнице без motion dataset.

Язык и конфигурация:

- Python 3.10+.
- Пакет устанавливается через `pip install -e .`.
- Конфигурация runtime построена на Hydra/OmegaConf и хранится в `teleopit/configs/`.
- Внутренние runtime-модули используют `InProcessBus` и протоколы из `teleopit/interfaces.py`.
- Sim2real изолирован по процессам в `teleopit/sim2real/mp/`.

## 4. Ключевые архитектурные границы

| Путь | Ответственность |
|---|---|
| `teleopit/interfaces.py` | Стабильные `Protocol`: Robot, Controller, InputProvider, Retargeter, ObservationBuilder и др. |
| `teleopit/pipeline.py` | Тонкий facade для sim runtime. |
| `teleopit/runtime/` | Разбор конфигов, разрешение путей, фабрики компонентов, CLI helpers. |
| `teleopit/bus/` | In-process zero-copy pub/sub. |
| `teleopit/inputs/` | BVH, UDP BVH, Pico4, video и преобразования входных координат. |
| `teleopit/retargeting/` | GMR и извлечение mimic reference. |
| `teleopit/controllers/observation.py` | Публичный `VelCmdObservationBuilder`. |
| `teleopit/controllers/rl_policy.py` | ONNX inference и fail-fast проверка размерностей. |
| `teleopit/robots/mujoco_robot.py` | MuJoCo-обёртка робота. |
| `teleopit/sim/` | SimulationLoop, reference timeline и runtime-компоненты. |
| `teleopit/sim2real/mp/` | Процессно-изолированная state machine, IPC и control loop реального G1. |
| `teleopit/sim2real/hands/` | Опциональное управление LinkerHand. |
| `teleopit/recording/` | HDF5-запись Pico sim2real. |
| `train_mimic/app.py` | Общая сборка train/play/benchmark. |
| `train_mimic/tasks/tracking/config/` | Регистрация tracking и ladder задач, env и PPO-конфиги. |
| `train_mimic/tasks/tracking/mdp/ladder.py` | Ladder FSM, targets, grip, rewards, curriculum и observations. |
| `train_mimic/tasks/tracking/rl/` | TemporalCNN и runners. |
| `train_mimic/data/` | Dataset builder, MotionLib, FK и конвертация. |

## 5. Неподлежащие нарушению инварианты

### Каноническая модель G1

Во всех актуальных путях — training, sim2sim, retargeting, dataset FK и ladder assembly — использовать только:

```text
assets/robots/unitree_g1/g1_29dof.xml
```

Этот XML и meshes скачиваются как внешний asset и не отслеживаются Git. Не создавать второй G1 XML и не копировать mesh tree для новой функциональности.

В репозитории всё ещё есть старые/экспериментальные tracked-файлы под `assets/ladder_g1/` и `show_scene.py`, но активная ladder-задача на них не должна опираться. Текущий код собирает лестницу поверх канонического XML во время конфигурации.

### Частоты управления

- Policy: 50 Hz.
- PD: 200 Hz.
- `decimation=4`, `sim_dt=0.005`.
- `realtime=true` всегда включает wall-clock pacing даже без viewer.
- `num_steps=0` означает бесконечный цикл.

### Преобразование action

RL policy выдаёт смещения относительно стандартной стойки:

```text
target_dof_pos = clip(action, -10, 10) * action_scale + default_dof_pos
```

`default_dof_pos` берётся из `teleopit/configs/robot/g1.yaml` (`default_angles`). `TeleopPipeline` обязан прокидывать его в controller config. Без этого колени и локти теряют standing offset, и G1 не удерживает баланс.

### Fail-fast вместо скрытого восстановления

- Несовпадение observation contract и ONNX signature должно завершать запуск с понятной ошибкой.
- Нельзя молча pad/trim/clip/replace логически неверные данные или конфиги «чтобы запустилось».
- Ошибка должна называть конфликтующие компоненты и прямой путь исправления.

## 6. Inference contract

Публичный observation: `velcmd_history`, текущий кадр имеет 167 измерений:

```text
ref_joint_pos(29)
+ ref_joint_vel(29)
+ ref_anchor_ori_b(6)
+ robot_base_ang_vel_b(3)
+ robot_joint_pos_rel(29)
+ robot_joint_vel(29)
+ prev_action(29)
+ robot_projected_gravity_b(3)
+ ref_anchor_lin_vel_b(3)
+ ref_anchor_ang_vel_b(3)
+ ref_projected_gravity_b(3)
+ ref_anchor_height(1)
= 167
```

`RLPolicyController` поддерживает dual-input ONNX:

- `obs`: текущий 167D кадр.
- `obs_history`: temporal history для TemporalCNN.

Экспорт tracking policy выполняется через `train_mimic/scripts/save_onnx.py`. Ladder policy имеет другой multi-group contract и не относится к текущему inference/export surface.

## 7. Runtime: sim2sim, sim2real и viewers

### Viewer contract

Единственный поддерживаемый ключ — `viewers`; старый alias `viewer` удалён.

- `sim2sim`: результат MuJoCo physics.
- `retarget`: кинематический retarget result.
- `mocap`: входной skeleton через MuJoCo custom geoms.
- `camera`: фиксированная RGB-камера G1 `d435i_rgb`.
- `all`: `mocap`, `retarget`, `sim2sim`; `camera` добавляется отдельно.
- `none`: без окон.

Окна viewer работают в отдельных subprocess, потому что GLFW/GLX не поддерживает несколько окон в одном процессе. Simulation завершается после закрытия всех активных окон.

Sim2real по умолчанию использует `viewers=none`; разрешён опциональный `viewers=retarget`.

### Offline BVH

- Sim2sim и стандартный sim2real читают `input.bvh_file` напрямую; отдельный UDP relay для offline playback не используется.
- Индекс BVH определяется временем: `int(policy_time * input_fps)`.
- Опциональное keyboard playback в sim2sim: `Space/P` pause/resume, `R` replay, `Q` stop.
- Pause удерживает командуемую позу.
- Resume сбрасывает policy/reference state и заново привязывает yaw/XY, без qpos interpolation и без сброса warm-start IK retargeter.
- `playback.pause_on_end=true` удерживает последний pose до ручного replay.
- На Unitree remote: `Start` -> `STANDING`, `Y` -> playback, `X` -> назад в `STANDING`, `L1+R1` -> `DAMPING`.

### Realtime reference timeline

Общий realtime-путь использует retargeted-reference timeline до построения observation.

Основные настройки:

- `retarget_buffer_enabled`
- `retarget_buffer_window_s`
- `retarget_buffer_delay_s`
- `reference_steps`, production default `[0]`
- `realtime_buffer_warmup_steps`
- `reference_velocity_smoothing_alpha`
- `reference_anchor_velocity_smoothing_alpha`

Pause/resume и переходы режимов сбрасывают policy/reference alignment и заново привязывают yaw/XY. Для Pico используется soft reset: GMR IK warm-start сохраняется.

## 8. Pico 4 realtime

- `Pico4InputProvider` читает tracking из in-process `pico_bridge.PicoBridge`.
- Поддерживаемая версия `pico-bridge`: `0.2.1`.
- Receiver работает на том же Teleopit host; отдельный onboard Pico input mode не нужен.
- Bone mapping: `pico_bridge_to_g1.json`.
- Input-space transform нельзя превращать в жёсткий публичный coordinate-system contract; после обновлений SDK/firmware его нужно проверять по retarget/sim2sim поведению.
- Video preview опционален и выключен по умолчанию. В sim2sim источник — MuJoCo `d435i_rgb`, в sim2real — RealSense.
- Отправка preview в Pico: `PicoBridge(video="frames").push_video_frame(rgb_uint8)`.

Pico sim2sim state machine:

```text
STANDING -> MOCAP <-> ARMS
```

Клавиши по умолчанию:

- `Y`: войти в `MOCAP`.
- `A`: pause/resume mocap.
- `B`: переключить `MOCAP`/`ARMS`.
- `X`: вернуться в `STANDING`.
- `Q`: выход.

В `ARMS` retargeting продолжает работать, но policy получает standing reference для body/legs/waist и live reference только для рук. При входе/выходе и resume сбрасывается alignment и применяется Kp ramp.

Sim2real reference worker:

- Во время `STANDING` и `DAMPING` он disarmed.
- При `STANDING -> MOCAP` worker rearm/reset происходит до принятия свежих references.
- Pause — состояние mocap session (`ACTIVE <-> PAUSED`), а не переход обратно в `STANDING`.

## 9. LinkerHand и запись sim2real

### LinkerHand

Включение:

```text
hands.enabled=true
hands.driver=linkerhand_l6|linkerhand_o6
hands.mode=gripper|vr_hand_pose
```

- По умолчанию выключено.
- `gripper` использует controller snapshot Pico grip/trigger.
- `vr_hand_pose` использует hand snapshot и `somehand==0.2.0` public API.
- Не запускать второй `PicoBridge` для рук.
- Teleopit сам конвертирует Pico 26-joint hand state в 21 landmarks; не импортировать `somehand.pico_input`.
- O6 поддерживает только `gripper`.
- O6 close pose: `[86, 73, 118, 111, 110, 111]`.
- L6 gripper speed по умолчанию `[50]*6`; O6 `[255]*6`; L6 `vr_hand_pose` всегда `[255]*6`.
- Low-latency somehand defaults: 60 Hz, 12 iterations, temporal/output alpha `1.0`.
- Hand control активен во всех sim2real modes. На shutdown или ошибке hand runtime отправляется open pose.
- При пропаже `vr_hand_pose` последняя команда удерживается, а не заменяется открытой рукой.

### HDF5 recording

Запись Pico sim2real включается через `--config-name sim2real_record` или `recording.enabled=true` и требует:

- `input.provider=pico4`;
- `input.video.enabled=true`;
- `input.video.source=realsense`;
- интерактивный terminal;
- optional dependency `recording`.

Управление: `R` start episode, `S` save, `D` discard, `Q` shutdown.

Сохраняются:

- `observation.images.d435i_rgb` как сжатый MP4 sidecar;
- `observation.state(68)`;
- `observation.mode(1)`;
- `action(36)`;
- `action.hand(12)`;
- `frame_index`, timestamps и video sync attributes в HDF5.

Raw RGB datasets внутри HDF5 не поддерживаются.

## 10. Motion tracking training

Задача: `General-Tracking-G1`, experiment: `g1_general_tracking`.

- TemporalCNN actor/critic с MLP dims `(2048, 1024, 512, 256, 128)`.
- Observation `velcmd_history`, 167D current frame, dual-input ONNX export.
- `window_steps=[0]`.
- Training sampling по умолчанию `rewind`; поддерживаются `uniform`, `start`, `rewind`.
- Play/benchmark включают `play=True`, что переключает sampling на `start`.
- `rewind` после failure с вероятностью `rewind_prob` возвращается в тот же clip на `rewind_min_steps..rewind_max_steps`, иначе делает uniform sampling.
- Rewards охватывают root/body pose и velocity, joints, survival, action rate, joint limits, self-collision и ankle acceleration.

Tracking-only entry points:

- `train_mimic/scripts/train.py`
- `train_mimic/scripts/benchmark.py`
- `train_mimic/scripts/save_onnx.py`

`play.py` теперь принимает и tracking, и ladder task.

## 11. RL-only ladder task — текущий главный рабочий блок

Задача: `G1-Ladder-Climb-RL`, experiment: `g1_ladder_rl`.

### Общая модель

- Только PPO через `train_mimic/scripts/train_ladder.py`.
- Motion dataset, GMR, imitation rewards и `MotionTrackingOnPolicyRunner` не используются.
- Action остаётся стандартным 29D G1 joint-position action.
- TemporalCNN actor/critic с отдельными Conv1d encoders для state history и ladder geometry, затем MLP `(2048, 1024, 512, 256, 128)`.
- Для ladder-конфигурации effort limits плеч, локтей и запястий уменьшены до 70% (`25 -> 17.5 Nm`, `5 -> 3.5 Nm`). Tracking/inference robot config не меняется.

### Геометрия и контакты

- A-frame ladder, grip sites и weld anchors генерируются поверх канонического `g1_29dof.xml` при конфигурации.
- На каждой стороне есть collidable side rails и отдельные flat-topped box rungs с открытыми промежутками.
- За каждой стороной расположен тонкий невидимый blocker. Он сталкивается только с pelvis/torso/head, но пропускает руки и ноги к rung bars.
- Default G1 collision editor отключает generated geoms без `.*_collision`; ladder config обязан повторно включать rails/rungs, сохранить stiff `condim=4` и отдельный collision bit для blocker/trunk.
- Руки крепятся к лестнице через weld environment mechanic; ноги используют только физические контакты без weld.
- Scene удаляет embedded `floor`, использует одну solid-color non-reflective plane, удаляет embedded lights и отключает playback shadows/reflections.

### Начальное состояние и FSM

- Эпизоды стартуют сразу в climbing keyframe: обе ноги физически на rung 2, обе руки attached к rung 5. Ground approach отсутствует.
- Пять публичных фаз повторяются в фиксированном порядке:

```text
STABILIZE -> FIRST_HAND -> SECOND_HAND -> FIRST_FOOT -> SECOND_FOOT
```

- Двигаться может только выбранная фазой конечность.
- Обе руки attached в течение foot phases.
- Hand target требует 3 последовательных valid frames.
- Foot target требует 5 последовательных valid frames и реального contact.
- Финальный success требует последнего hand rung и согласованной final foot support; fall и low-root — failure.

### Feedback-controlled PRE_RELEASE

Каждая hand phase начинается внутренним `PRE_RELEASE`, который не добавляет шестую публичную фазу:

1. 8 стабильных preload frames при сохранённом weld.
2. 20-frame smoothstep ослабление индивидуального weld от compiled `solref/solimp` до `timeconst=0.18`, impedance `0.05`.
3. При потере support/torso/COM gate ramp откатывается на 2 шага.
4. Joint motion penalized, но не блокирует release.
5. После полного ослабления нужны ещё 5 stable frames.
6. Только затем weld бинарно отключается.
7. Более 300 policy steps в `PRE_RELEASE` завершает episode как failure с общим penalty `-50`.

`eq_solref` и `eq_solimp` разворачиваются per environment, поэтому параллельные среды ослабляют weld независимо. `grip_strength` меняется непрерывно и используется как часть command, rung markers и support centroid.

### Stabilization gates

`STABILIZE` требует случайный per-environment continuous dwell от 50 до 100 policy frames. Потеря любого условия сбрасывает dwell:

- обе ноги имеют physical support;
- обе руки attached;
- torso COM speed `<= 0.20 m/s`;
- joint-speed RMS `<= 1.0 rad/s`;
- max torso/pelvis angular speed `<= 0.40 rad/s`;
- max absolute waist-joint speed `<= 0.60 rad/s`;
- support-relative torso offset `<= 0.18 m`.

Torso orientation — soft shaping/diagnostic, а не direct phase-transition gate. При этом отдельный orientation gate может участвовать во внутренней логике безопасного PRE_RELEASE.

### Observation contract ladder policy

- Base current actor group: 117D.
- Base clean critic group: 120D.
- Actor и critic получают собственную 10-frame history.
- 24D ladder command содержит five-state phase one-hot, torso-frame hand/foot targets, hand attachment, active-hand grip strength, physical foot contacts и normalized hand/foot progress.
- Отдельная geometry input: `9 x 15` torso-frame rung tokens. Каждый token содержит finite endpoints, validity и target/support markers для конечностей.
- Только critic получает дополнительную current-only 14D privileged reward/FSM group; она намеренно не входит в `critic_history`.
- 14D group включает body/record heights, cycle/phase ascent, torso/joint stability, два physical foot supports, required-support validity, release progress и dwell progress.

### Adaptive prefix curriculum и reset bank

- Curriculum использует окно последних 100 eligible terminal episodes текущей фазы.
- Promotion происходит только при success rate строго `> 80%` и после минимального числа environment steps.
- Для initial stabilization минимум 36,000 environment steps, то есть 1,500 PPO iterations при rollout length 24.
- Для каждого последующего перехода минимум 120,000 steps, то есть 5,000 iterations.
- Полный цикл не может разблокироваться раньше iteration 16,500.
- Достижение locked boundary завершает короткий prefix episode без failure penalty.
- Boundary-started episodes обучают PPO, но исключаются из promotion window.
- Успешные transitions записывают GPU reset bank per phase: root/joint state, hand anchors, weld state, hand/foot rung assignments.
- После появления bank 50% resets берут boundary state, остальные 50% используют deterministic climbing start.
- Выбирать можно только уже unlocked phase banks; FSM phase задаётся после физического восстановления состояния.
- `LadderOnPolicyRunner` объединяет outcomes со всех distributed ranks раз за PPO iteration.
- Checkpoint сохраняет unlocked phase, phase-start step, recent/pending outcomes и boundary state bank.
- Adaptive resume обязан fail-fast, если checkpoint не содержит совместимого curriculum state.

### Reward contract

Текущий reward set фиксирован и не имеет schedule:

| Term | Weight |
|---|---:|
| Supported novel maximum whole-body height | `20` |
| Signed phase-aware physical foot placement | `8` |
| Phase target/release progress | `8` |
| Любое ordered phase completion | `25` |
| Stabilization-only torso-orientation squared error | `-1` |
| Unsuccessful termination | `-50` |
| Final success | `100` |
| Action rate | `-0.05` |
| Joint acceleration | `-3e-7` |
| Joint limits | `-5` |

Важные свойства:

- Нет отдельного dense support, COM, upright, survival, joint-velocity, limb-progress, torso-ascent или cycle-completion reward.
- Height reward использует среднюю высоту pelvis/torso COM, платит только за новый episode maximum и только при необходимых физических supports.
- Unsupported jump всё равно продвигает record, но не получает reward; lowering/re-climbing не фармит награду.
- Foot placement — signed potential difference с `distance_std=0.06`: неизменный contact даёт 0, потеря и восстановление симметричны.
- Phase potential использует normalized target distance и `1 - grip_strength` в hand phases.
- Ослабление и обратный regrip симметричны; замкнутый цикл имеет нулевую сумму.
- В `STABILIZE` оплачивается только рост per-phase maximum normalized dwell, поэтому сброс и повтор того же dwell не фармит reward.
- Positive target progress при временно invalid support/posture обычно сохраняет 25% веса; после detach руки positive hand progress платится только при valid constraints.
- Negative progress не attenuated.
- Успех и корректно завершённый locked curriculum prefix исключены из termination penalty. Timeout, fall, low-root и stalled `PRE_RELEASE` получают `-50`.

Movement phase completion требует:

- support-relative COM offset `<= 0.15 m`;
- torso speed `<= 0.20 m/s`;
- joint-speed RMS `<= 1.0 rad/s`.

Дополнительно:

- First foot не может опустить whole-body height более чем на `0.03 m` от начала фазы.
- Second foot требует минимум `0.12 m` whole-body ascent от начала цикла.
- Torso orientation не блокирует phase completion.

### PPO exploration

- Initial Gaussian action std: `0.7`.
- Effective range: `[0.25, 1.0]`.
- Runner проектирует raw scalar/log std обратно в нативные границы после PPO update и checkpoint load, чтобы clamp не создавал gradient-dead parameter.
- Логируется `Policy/raw_mean_std`.
- При каждом curriculum promotion std сбрасывается на `0.7`, optimizer state для параметра очищается, adaptive learning rate и optimizer groups возвращаются к `5e-4`.

### Playback, benchmark и video

- `play.py --task G1-Ladder-Climb-RL` запускает ladder без motion dataset.
- `--ladder_phase stabilize|first_hand|second_hand|first_foot|second_foot` выбирает самый глубокий разрешённый ordered prefix; default `second_foot` означает полный climb.
- Общая логика выбора вынесена в новый `train_mimic/ladder_playback.py`.
- `benchmark_ladder.py` не обновляет PPO, считает success/failure/timeout и может записывать MP4 для одной среды.
- `record_ladder_video.py` пишет только rollout video без benchmark report.
- При явном `--ladder_phase` recorder замораживает успешно завершённую выбранную фазу без перехода и без curriculum-boundary termination.
- Без `--ladder_phase` recorder снимает полный climb.
- Recorder останавливается до следующего автоматического reset и использует fixed world-space overview вне лестницы.

### Несовместимость checkpoint

Полный resume несовместим с checkpoint, созданными до текущего контракта, включая:

- старые 105D/108D contracts;
- flat 117D/120D MLP;
- TemporalCNN с `9 x 7` endpoint-only geometry;
- отсутствие critic-only 14D group;
- старую семантику command без continuous grip strength;
- старую scheduled multi-term reward system;
- fixed-schedule curriculum без текущего state/reset bank;
- версии до feedback soft release и новых ascent gates.

Для текущего дизайна нужна новая тренировка с нуля. Если старый actor всё же используется, это должен быть явный actor-only warm start с заново инициализированным critic.

## 12. Dataset pipeline

Pipeline намеренно разделён на minimal distribution format и precomputed training format.

### Build

- Spec поддерживает `preprocess`: root-XY normalization, ground alignment и basic clip filtering.
- `build_dataset.py` создаёт minimal HDF5 shards в `data/datasets/<dataset>/`.
- Нет train/val split и manifest.
- Shard discovery recursive.
- Minimal shard хранит только `root_pos`, `root_quat_w`, `joint_pos`, `body_names`, `clip_starts`, `clip_lengths`, `clip_fps`.
- Длинные clips режутся на bounded overlapping windows.
- `build_dataset.py` не должен запускать precompute.

### Precompute и training

- `precompute_dataset.py` создаёт отдельный precomputed training directory.
- `motion_file` training обязан указывать на precomputed dataset, а не на minimal distributed shards.
- Joint velocities и body FK/velocities читаются из precomputed shards; MuJoCo FK не должен выполняться при загрузке clips.
- `MotionLib` загружает все найденные windows в CPU/GPU memory на старте.
- Sampling идёт только по valid center frames для `window_steps`; default `[0]`.

### Pico motion dataset

- `scripts/run/record_pico_motion.py` записывает live Pico body tracking как retargeted G1 NPZ в `data/pico_motion/clips/`.
- Viewer: live `Retarget`.
- Terminal keys: `R/S/D/N/Q`.
- Semantic label хранится в filename; отдельный per-clip JSON намеренно не создаётся.
- Сборка recorded clips: `python train_mimic/scripts/data/build_dataset.py --spec data/pico_motion/pico_recorded.yaml --force`.

## 13. GMR и BVH

- GMR self-contained в `teleopit/retargeting/gmr/`.
- Нужны внешние groups `robots` и `gmr`.
- Поддержан `lafan1`: 22 joints, 30 FPS, centimeters.
- Поддержан `hc_mocap`: 50 joints, 60 FPS с downsample до 30 FPS, meters.
- `lafan1-resolved` пока не поддержан: нужен отдельный skeleton adapter.

IK offset для пары `(robot_body, human_bone)`:

```text
R_result = R_human * R_offset
R_offset = inverse(R_human_tpose) * R_robot_tpose
```

Quaternion order: `(w, x, y, z)`. До вычисления нужно совместить forward direction robot root и BVH human. Для `hc_mocap` G1 смотрит `+X`, human — `-Y`, поэтому robot root получает `-90 deg` вокруг Z. Инструмент: `scripts/dev/compute_ik_offsets.py`.

## 14. Dependencies и окружение

Основные dependencies перечислены в `pyproject.toml`: MuJoCo, mink, qpsolvers, torch, scipy, numpy, Hydra/OmegaConf, h5py, onnxruntime и др.

Extras:

- `dev`: pytest/coverage.
- `sim2real`: OpenCV; G1 bridge устанавливается отдельно.
- `train`: `torch>=2.7`, `rsl-rl-lib==5.2.0`, `mjlab==1.4.0`, `warp-lang==1.15.0`, wandb, swanlab.
- `pico4`: `pico-bridge[camera]==0.2.1` из release wheel и sim2real dependencies.
- `recording`: Pico4, OpenCV, imageio/ffmpeg.

Критический pin:

```text
mjlab==1.4.0
warp-lang==1.15.0
```

`warp-lang==1.16.0` несовместим с поддерживаемым MuJoCo Warp 3.8 stack и падает при sensor-kernel code generation с `Referencing undefined symbol: xmat`. Все training/playback entry points, импортирующие training stack, должны проверять версию Warp до создания CUDA environment.

## 15. Внешние assets

Не коммитить robot meshes, datasets, checkpoints, GMR assets и demo media.

Primary ModelScope repositories:

| Repo | Тип | Содержимое |
|---|---|---|
| `BingqianWu/Teleopit-models` | model | checkpoints, GMR assets, sample BVH, robot archives |
| `BingqianWu/Teleopit-datasets` | dataset | dataset shards |

Alternative HuggingFace repositories:

- `12e21/Teleopit-models`
- `12e21/Teleopit-datasets`

Старый `BingqianWu/Teleopit-assets` deprecated; не загружать туда новые releases.

Asset groups:

- `ckpt`: `track.onnx`, `track.pt`.
- `robots`: canonical `assets/robots/` archive.
- `gmr`: `teleopit/retargeting/gmr/assets/` archive.
- `bvh`: sample BVH.
- `data`: dataset repository.

Перед push запускать `python scripts/dev/check_large_tracked_files.py`.

## 16. Основные команды

### Установка и assets

```bash
pip install -e .
pip install modelscope
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
```

Training environment:

```bash
pip install -e '.[train]'
python -c "import warp as wp; print(wp.__version__)"
```

### Offline sim2sim

```bash
python scripts/run/run_sim.py \
  controller.policy_path=track.onnx \
  input.bvh_file=data/sample_bvh/aiming1_subject1.bvh

python scripts/run/run_sim.py \
  controller.policy_path=track.onnx \
  input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
  'viewers=[mocap,retarget,sim2sim,camera]'
```

### Pico sim2sim / sim2real

```bash
python scripts/run/run_sim.py \
  --config-name pico4_sim \
  controller.policy_path=track.onnx

python scripts/run/run_sim2real.py \
  --config-name pico4_sim2real \
  controller.policy_path=track.onnx
```

### Offline BVH sim2real

```bash
python scripts/run/run_sim2real.py \
  controller.policy_path=track.onnx \
  input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

### Dataset

```bash
python train_mimic/scripts/data/build_dataset.py \
  --spec train_mimic/configs/datasets/twist2.yaml

python train_mimic/scripts/data/precompute_dataset.py \
  data/datasets/twist2 \
  --outdir data/datasets/twist2_precomputed \
  --jobs 8 \
  --force
```

### Tracking train / play / export

```bash
python train_mimic/scripts/train.py \
  --motion_file data/datasets_precomputed

python train_mimic/scripts/play.py \
  --task General-Tracking-G1 \
  --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
  --motion_file data/datasets_precomputed

python train_mimic/scripts/save_onnx.py \
  --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
  --output policy.onnx \
  --history_length 10
```

### Ladder train / play / benchmark / video

```bash
python train_mimic/scripts/train_ladder.py \
  --num_envs 4096 \
  --max_iterations 60000

python train_mimic/scripts/play.py \
  --task G1-Ladder-Climb-RL \
  --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
  --ladder_phase second_hand

python train_mimic/scripts/benchmark_ladder.py \
  --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
  --num_envs 64 \
  --num_eval_steps 5000

python train_mimic/scripts/record_ladder_video.py \
  --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
  --output ladder.mp4 \
  --frames 1000 \
  --ladder_phase second_hand
```

Multi-GPU ladder training поддерживает `--gpu_ids 0 1 2 3`.

## 17. Проверка и текущее состояние тестов

Нормальная команда проекта:

```bash
python -m pytest tests/ -v
```

На момент создания handoff в текущем Windows/Anaconda environment тесты не были фактически выполнены:

- `pytest -q` остановился на `ModuleNotFoundError: teleopit` из-за способа запуска/окружения.
- `python -m pytest -q` дошёл до collection, но получил 29 collection errors из-за отсутствующих dependencies, включая `torch`, `mjlab`, `mink`, `omegaconf`, `viser` и `g1_bridge_sdk`.
- Это не свидетельствует о падении самих тестов; текущая среда просто не является установленным dev/train environment.
- Синтаксическая компиляция всех изменённых Python-файлов ladder-блока через `python -m py_compile` завершилась успешно.

После восстановления корректного окружения в первую очередь проверить:

```bash
python -m pytest \
  tests/test_ladder_training.py \
  tests/test_play_script.py \
  tests/test_record_ladder_video.py \
  tests/test_runner_iteration_numbering.py -v
```

Затем выполнить весь `tests/`, short ladder environment check (`--num_envs 64 --max_iterations 100`) и хотя бы один playback/recording smoke test с совместимым checkpoint.

## 18. Что именно сделано в текущем незакоммиченном блоке

Крупный diff, который нужно сохранить и довести до проверки, включает:

- переработанный ladder command/FSM;
- feedback-controlled per-environment weld soft release;
- continuous grip strength;
- random stabilization dwell и расширенные gates/metrics;
- target/support-aware `9 x 15` rung tokens;
- critic-only 14D privileged reward/FSM observation;
- phase-boundary GPU reset bank и checkpoint serialization;
- новый fixed reward contract без schedule;
- protected PPO std projection и reset на curriculum promotion;
- ladder-specific reduced upper-body effort limits;
- обработку episode metrics, появляющихся не на первом reset внутри rollout;
- поддержку ladder task в общем `play.py`;
- общий helper `train_mimic/ladder_playback.py`;
- `--ladder_phase` для play и recorder;
- синхронные изменения README, English docs и Chinese translations;
- большой набор unit tests для новых semantics.

Размер tracked diff до добавления этого файла: примерно 5,049 insertions и 1,354 deletions. Большая часть приходится на `ladder.py`, `env.py` и `test_ladder_training.py`.

## 19. Известные проблемы и риски

1. `lafan1-resolved` требует отдельного adapter и остаётся сломанным.
2. Старые GMR XML под `teleopit/retargeting/gmr/assets/unitree_g1/` не являются entry point.
3. В корне/`assets/ladder_g1/` есть legacy experimental ladder assets; не переносить архитектуру обратно на них.
4. Текущий ladder diff пока не подтверждён тестами в этой среде из-за отсутствующих dependencies.
5. Старые ladder checkpoints несовместимы с текущим observation/reward/curriculum contract.
6. Assets, datasets, logs, benchmark results и checkpoints в основном gitignored и могут отсутствовать после clone/переноса.
7. Для sim2real нужны физический G1, корректная network interface, отдельно установленный `g1_bridge_sdk` и соблюдение safety state machine. Не ослаблять проверки ради удобства запуска.

## 20. Правила дальнейшей работы для ChatGPT

- Перед правками читать `AGENTS.md` и локальные инструкции, если они появятся глубже в дереве.
- Сохранять модульные границы и существующие публичные интерфейсы.
- Не менять одновременно tracking и ladder contracts без явной необходимости; ladder-specific параметры не должны влиять на standard tracking/inference G1.
- Любое изменение observation contract должно синхронно отражаться в env config, model inputs, checkpoint compatibility, tests и docs.
- Любое изменение ladder FSM/reward/curriculum должно проверяться на reward farming, dt-dependence, multi-env independence, distributed synchronization и resume compatibility.
- После крупных features обновлять вместе `AGENTS.md`, `README.md`, English docs и соответствующие Chinese translations.
- Chinese docs всегда переводить из уже обновлённого English source; не развивать их независимо.
- User-facing docs должны описывать стабильный продуктовый контракт, а не историю конкретного patch.
- Не добавлять крупные binaries в Git. Перед push выполнять large-file check.
- Не делать auto-commit и не очищать dirty worktree без прямого запроса пользователя.

## 21. Рекомендуемый порядок продолжения

1. Создать корректное dev/train окружение с pinned `mjlab==1.4.0` и `warp-lang==1.15.0`.
2. Запустить targeted ladder tests из раздела 17.
3. Исправить только реальные failures, сохраняя описанные contracts.
4. Запустить полный `tests/` в окружении со всеми нужными extras либо разделить hardware-only collection от обычных unit tests.
5. Провести short ladder training smoke test.
6. Проверить `play.py` для tracking и ladder, затем recorder с явным phase prefix и без него.
7. Проверить, что README, English docs, Chinese docs и `AGENTS.md` всё ещё согласованы с кодом.
8. Выполнить `python scripts/dev/check_large_tracked_files.py`.
9. Перед коммитом показать пользователю итоговый diff и результаты проверок; не коммитить автоматически.
