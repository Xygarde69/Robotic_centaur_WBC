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
centaur_rl/tasks/velocity/config/centaur/               env + PPO config (Centaur-Flat / Centaur-Rough)
centaur_rl/tasks/velocity/{velocity_env_cfg.py,mdp,rl}  copied from unitree_rl_mjlab
scripts/{train,play,list_envs}.py                       copied from unitree_rl_mjlab
```

Needs `mjlab==1.2.0` / `mujoco-warp==3.5.0` (e.g. the unitree_rl_mjlab venv):

```bash
source ~/Documents/Projects/repo/unitree_rl_mjlab/.venv/bin/activate
python mjcf/build_train_xml.py            # only after editing centaur.xml / upper-body pose
cd scripts
python train.py Centaur-Flat  --env.scene.num-envs=4096
python train.py Centaur-Rough --env.scene.num-envs=1024
python train.py Centaur-Rough-SGS --env.scene.num-envs=1024   # Success-Guided Sampling (paper)
python play.py  Centaur-Flat  --checkpoint_file=logs/rsl_rl/centaur_velocity/<run>/model_<iter>.pt
```

Notes: forward (+x) is the torso end. Only the calf meshes collide. Effort limit 100 Nm.
`*-SGS` tasks implement Success-Guided Sampling (2610.12465v1.pdf) in
`centaur_rl/tasks/velocity/mdp/sgs.py` and replace the terrain/command curricula.
See `log.txt` for every change and the reason behind it.
