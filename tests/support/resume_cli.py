"""Runs or resumes the run that `tests.support.resumable` configures; prints a JSON line when it ends."""

import json
import sys

import auxein
from tests.support.resumable import settings


def main() -> None:
    config = json.loads(sys.argv[1])
    entry = auxein.resume if config.get("resume") else auxein.run
    result = entry(budget=auxein.Budget(evaluations=config["evaluations"]), **settings(config))
    print(json.dumps({"evaluations": result.evaluations_used, "stop": result.stop_reason}))


if __name__ == "__main__":
    main()
