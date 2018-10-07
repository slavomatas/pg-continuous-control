import numpy as np

from utils import *


class BaseTask:
    def __init__(self):
        pass

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done:
            next_state = self.env.reset()
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)


class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir=None, single_process=False):
        if single_process:
            self.tasks = [task_fn(log_dir=log_dir) for _ in range(num_workers)]
        else:
            self.tasks = [ProcessTask(task_fn, log_dir) for _ in range(num_workers)]
        self.state_dim = self.tasks[0].state_dim
        self.action_dim = self.tasks[0].action_dim
        self.name = self.tasks[0].name
        self.single_process = single_process

    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return np.stack(results)

    def close(self):
        if self.single_process:
            return
        for task in self.tasks: task.close()