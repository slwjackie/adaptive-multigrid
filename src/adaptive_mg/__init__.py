from .config import MGConfig
from .checkpoint import TemporalComponents
from .solver import PreparedTemporalMGSolver, PreparedNeuralMGSolver, SolveResult, solve
from .pde import DiffusionCase, assemble_stiffness, case_suite
from .strategy import STRATEGIES, get_strategy
__version__='0.6.7'
__all__=['MGConfig','TemporalComponents','PreparedTemporalMGSolver','PreparedNeuralMGSolver','SolveResult','solve','DiffusionCase','assemble_stiffness','case_suite','STRATEGIES','get_strategy']

# New production API. The old exact-K names above remain legacy-compatible.
from .v67 import AdaptiveConfig, Components, PreparedAdaptiveMG, AdaptiveResult
__all__ += ['AdaptiveConfig','Components','PreparedAdaptiveMG','AdaptiveResult']
