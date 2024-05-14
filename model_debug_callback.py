from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        print(self.locals["info"])
        return True