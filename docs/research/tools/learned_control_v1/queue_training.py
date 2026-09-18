"""Launch a prospectively recorded recipe once its atomic dataset is available."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from docs.research.tools.learned_control_v1 import training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    dataset = Path(recipe['dataset'])
    dependency = dataset.with_suffix('.json')
    deadline = datetime.fromisoformat(recipe['deadline_utc'])
    while not dependency.exists():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Dataset did not arrive before registered training deadline')
        failure = dataset.parent / 'failure.json'
        if failure.exists():
            raise RuntimeError('Demonstration collection failed: ' + failure.read_text())
        time.sleep(10)
    plan = training.freeze_plan(args.root, recipe)
    print(json.dumps({'plan_sha256': training.sha(plan), 'ready_at': training.now()}), flush=True)
    print(json.dumps(training.train(args.root)), flush=True)


if __name__ == '__main__':
    main()
