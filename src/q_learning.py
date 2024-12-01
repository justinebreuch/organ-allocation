from dataclasses import dataclass, field
import time
import numpy as np
from typing import List, Tuple
import os
import pandas as pd
import logging
from tqdm import tqdm


@dataclass
class Environment:
    states: List[int] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)


@dataclass
class QLearning:
    max_iterations: int = 20
    environment: Environment = None
    discount_factor: float = 0.95
    learning_rate: float = 0.1
    input_file: str = field(default="")
    epsilon: float = 1.0

    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.environment:
            raise ValueError(
                f"Can't build Q(s,a). Must initialize environment")

        self.q = {
            (p, v): np.zeros(len(self.environment.actions))
            for p in range(len(self.environment.states) // 100)
            for v in range(len(self.environment.states) // 500)
        }

    @staticmethod
    def get_dataframe(infile: str) -> pd.DataFrame:
        df = pd.read_csv(infile)
        df = df.rename(
            columns={"s": "state", "a": "action", "r": "reward", "sp": "next_state"})
        return df

    @staticmethod
    def get_features(state: int) -> Tuple[int, int]:
        p = state % 500
        v = state // 500
        return p, v

    def update(self, state: int, action: int, reward: float, next_state: int):
        p, v = self.get_features(state)
        next_p, next_v = self.get_features(next_state)

        best_next_action = self.get_max_action(next_p, next_v)
        self.q[(p, v)][action] += self.learning_rate * (reward + (self.discount_factor *
                                                                  self.q[(next_p, next_v)][best_next_action]) - self.q[(p, v)][action])

    def get_max_action(self, p: int, v: int):
        return np.random.choice(np.flatnonzero(self.q[(p, v)] == self.q[(p, v)].max()))

    def learn(self):
        df = self.get_dataframe(self.input_file)
        df = df.sample(frac=1).reset_index(drop=True)

        for _ in tqdm(range(self.max_iterations)):
            for _, row in df.iterrows():
                self.update(
                    state=row['state']-1,
                    action=row['action']-1,
                    reward=row['reward'],
                    next_state=row['next_state']-1
                )
        self.write_to_file()
        return self.q

    def run(self):
        start_time = time.perf_counter()
        self.learn()
        elapsed_time = time.perf_counter() - start_time
        self.logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        self.write_to_file()
        return self.q

    def write_to_file(self):
        output_file = f'{os.path.splitext(self.input_file)[0]}.policy'
        with open(output_file, 'w') as f:
            output = []
            for state in self.environment.states:
                p, v = self.get_features(state-1)
                output.append(np.argmax(self.q[(p, v)]) + 1)
            f.write('\n'.join(map(str, output)))
