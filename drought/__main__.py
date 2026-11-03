"""Command line for the pipeline: python -m drought synth | run | demo"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import pipeline, synthetic


def cmd_synth(args):
    Path(args.path).parent.mkdir(parents=True, exist_ok=True)
    synthetic.write_cube(synthetic.make_cube(seed=args.seed), args.path)
    print(f'Wrote {args.path}')


def cmd_run(args):
    _, report = pipeline.run(args.input, args.outdir, map_month=args.map_month)
    print(json.dumps(report['summary'], indent=2))
    print(f'Outputs are in {args.outdir}')


def cmd_demo(args):
    """Plant a drought, run the pipeline blind, then see how much of it came back."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scene = outdir / 'scene.nc'
    synthetic.write_cube(synthetic.make_cube(seed=args.seed), scene)
    products, report = pipeline.run(scene, outdir, map_month='2021-08')

    classes = products['drought_class']
    truth = synthetic.drought_truth(classes['time'].values, classes['lat'].values, classes['lon'].values)
    valid = classes.values != -1
    scores = {
        'anomaly_classes': pipeline.detection_scores(classes.values >= 1, truth, valid),
        'vci_below_20': pipeline.detection_scores(np.nan_to_num(products['vci'].values, nan=100) < 20, truth, valid),
    }
    (outdir / 'demo_scores.json').write_text(json.dumps(scores, indent=2) + '\n')

    print(json.dumps(report['summary'], indent=2))
    print('How well the planted drought came back:')
    print(json.dumps(scores, indent=2))
    print(f'Outputs are in {outdir}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m drought', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    synth = commands.add_parser('synth', help='write a synthetic red/NIR cube')
    synth.add_argument('path')
    synth.add_argument('--seed', type=int, default=42)
    synth.set_defaults(func=cmd_synth)

    run = commands.add_parser('run', help='run the pipeline on a NetCDF red/NIR cube')
    run.add_argument('input')
    run.add_argument('--outdir', default='out')
    run.add_argument('--map-month', help='month for the class map, like 2021-08. Defaults to the last one')
    run.set_defaults(func=cmd_run)

    demo = commands.add_parser('demo', help='plant a drought, run the pipeline, score the result')
    demo.add_argument('--outdir', default='out')
    demo.add_argument('--seed', type=int, default=42)
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except pipeline.InvalidInput as error:
        print(f'Input failed validation: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
