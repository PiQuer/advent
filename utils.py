from typing import Sequence, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataSetBase:
    input_file: Path
    result: Any
    id: str

    def preprocess(self):
        return self.input_file.read_text().splitlines()


# noinspection PyArgumentList
def dataset_parametrization(day: str, examples: Sequence[tuple[str, Any]], result: Any,
                            dataset_class: type[DataSetBase] = DataSetBase):
    examples = [dataset_class(input_file=Path(f"input/day_{day}_example{example[0]}.txt"),
                              result=example[1],
                              id=f"example{example[0]}") for example in examples]
    puzzle = dataset_class(input_file=Path(f"input/day_{day}.txt"), result=result, id="puzzle")
    return {'argnames': "dataset", 'argvalues': examples + [puzzle], 'ids': lambda x: x.id}
