import multiprocessing
import gymnasium as gym


def worker_process(remote: multiprocessing.connection.Connection, seed: int, str_env: str):
    """
    Each worker process runs this method. It initializes a game environment and waits for instructions from the parent process through a multiprocessing connection.
    The worker process can handle commands such as "step", "reset", and "close" to interact with the game environment.

    Parameters:
    -----------
    remote : multiprocessing.connection.Connection
        The connection object used for communication between the parent and child processes.
    seed : int
        The seed value for the random number generator.
    str_env : str
        The name of the environment to create.

    Commands:
    ---------
    step : Executes a step in the game environment with the provided data and sends the result back.
    reset : Resets the game environment and sends the initial state back.
    close : Closes the connection and terminates the worker process.
    """

    # Create environment
    env = gym.make(str_env, render_mode="rgb_array")

    # Wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "reset":
            remote.send(env.reset(seed=seed))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    """
    The Worker class manages a separate process for performing tasks.
    It utilizes multiprocessing to create a child process that communicates with the parent process through a pipe.
    """

    def __init__(self, seed: int, str_env: str):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed, str_env))
        self.process.start()