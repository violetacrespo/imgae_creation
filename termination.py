from pymoo.core.termination import Termination


class CombinedTerminationOR(Termination):
    """Combine several termination conditions using logical OR."""

    def __init__(self, terminations):
        super().__init__()
        self.terminations = terminations

    def _update(self, algorithm):
        return any(t._update(algorithm) for t in self.terminations)
