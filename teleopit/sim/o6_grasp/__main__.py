"""Run with conda run -n teleopit python -m teleopit.sim.o6_grasp."""
import argparse
import json

from .config import load_config
from .evaluate import run_suite, run_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON config overrides')
    parser.add_argument('--output', default='outputs/o6_grasp')
    parser.add_argument('--seed', type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--suite', action='store_true')
    mode.add_argument('--viewer', action='store_true')
    mode.add_argument('--render', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.viewer:
        from .viewer import show
        show(config)
    elif args.render:
        from .viewer import render_frames
        render_frames(config, args.output)
    elif args.suite:
        results = run_suite(config, args.output, seed=args.seed)
        print(json.dumps([{k: row[k] for k in ('scenario', 'unsupported_success', 'verified_seconds', 'failure_observed', 'release_clear')}
                          for row in results], indent=2))
    else:
        print(json.dumps(run_trial('baseline', config, args.output, seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
