from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.core.termination import Termination


class CombinedTerminationOR(Termination):

    def __init__(self, terminations):
        super().__init__()
        self.terminations = terminations

    def _update(self, algorithm):
        return any(t._update(algorithm) for t in self.terminations)
