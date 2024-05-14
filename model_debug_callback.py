from stable_baselines3.common.callbacks import BaseCallback


class DebugCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # print(self.locals)
        return True
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print(self.locals["infos"])
        print(self.locals["rewards"])
        print("-------------------------------------")