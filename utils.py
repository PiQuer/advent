from typing import Sequence, Any
from pathlib import Path


def dataset_cases(day: str, examples: Sequence[tuple[str, Any]], result: Any):
    class Cases:
        pass

    Cases.puzzle = lambda self: (Path(f"input/day_{day}.txt"), result)
    for example in examples:
        setattr(Cases, f"example_{example[0]}",
                lambda self: (Path(f"input/day_{day}_example{example[0]}.txt", example[1])))
    return Cases
