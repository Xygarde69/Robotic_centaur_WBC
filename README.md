# Robotic_centaur_WBC
# Robotic_centaur_WBC

## Legged locomotion training (mjlab + rsl_rl PPO)

Same velocity-tracking paradigm as `unitree_rl_mjlab` (Go2 task), ported to the Centaur.
The training model is currently the **quadruped only**: the upper body (torso, neck, arms)
is removed (`UPPER_BODY_MODE` in `mjcf/build_train_xml.py`; "weld" restores it as a static payload).
The policy controls the 12 leg joints.

```
mjcf/centaur.xml            full CAD export (unchanged)
mjcf/build_train_xml.py     -> mjcf/centaur_train.xml (legs-only training model)
centaur_rl/assets/robots/centaur/centaur_constants.py   actuators, stance, collisions
centaur_rl/tasks/{velocity,trot,walking,pace}/config/centaur/   env + PPO config per task
centaur_rl/tasks/walking/                                   copy of velocity (Centaur-Walking-* tasks), to be adapted for walking
centaur_rl/tasks/pace/                                      copy of walking (Centaur-Pace-* tasks): pace gait with flight phases
centaur_rl/tasks/velocity/{velocity_env_cfg.py,mdp,rl}  copied from unitree_rl_mjlab (Centaur-Velocity-*)
scripts/{train,play,list_envs}.py                       copied from unitree_rl_mjlab
```

Needs `mjlab==1.2.0` / `mujoco-warp==3.5.0` (e.g. the unitree_rl_mjlab venv):

```bash
source ~/Documents/Projects/repo/unitree_rl_mjlab/.venv/bin/activate
python mjcf/build_train_xml.py            # only after editing centaur.xml / upper-body pose
cd scripts
python train.py --task Centaur-Velocity-Flat  --env.scene.num-envs=4096
python train.py --task Centaur-Trot-Rough --env.scene.num-envs=1024
python train.py --task Centaur-Trot-Rough-SGS --env.scene.num-envs=1024   # Success-Guided Sampling (paper)
python train.py --task Centaur-Pace-Rough-SGS --env.scene.num-envs=1024   # pace, SGS over tile x speed x turn
python play.py  Centaur-Trot-Flat  --checkpoint_file=logs/rsl_rl/centaur_trot/<run>/model_<iter>.pt
python train.py --help                       # lists every task
```

Notes: forward (+x) is the torso end. Only the calf meshes collide. Effort limit 100 Nm.
`*-Rough-SGS` tasks (SGS is rough-only; flat tasks use the plain curricula) implement Success-Guided Sampling (2610.12465v1.pdf) in
`centaur_rl/tasks/trot/mdp/sgs.py` and replace the terrain/command curricula.
See `log.txt` for every change and the reason behind it.
