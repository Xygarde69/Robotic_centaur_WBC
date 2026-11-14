"""Multi-gait pipeline: walk -> trot -> canter -> gallop, one policy per gait, straight + flat.

Each gait is trained in stages (see pipeline.py):
  ladder   SGS over forward-speed bins, mostly at the frontier speed V_current, which
           climbs from the gait's minimum to its maximum as each speed is mastered.
  refine   the same gait, speeds over the whole range (SGS-weighted), commands
           resampled mid-episode (up and down). Its checkpoint is the gait's policy.
  handoff  optional short fine-tune that starts episodes from states recorded from the
           neighbouring gaits' policies (upshift and downshift handoffs).

The next gait forks from a checkpoint of the previous gait's ladder (actor only: fresh
critic, fresh rewards, critic warmup with the actor frozen, std floor, observation
normalizer re-adaptation).

Entry point: `python train.py --task Multi-Gait_straight_flat_SGS` (orchestrator.py).
"""
