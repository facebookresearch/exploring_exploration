from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_habitat_sim():
    try:
        import habitat_sim

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        habitat_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
        from habitat.sims.habitat_simulator.action_spaces import (
            HabitatSimV1ActionSpaceConfiguration,
        )
    else:

        @registry.register_simulator(name="Sim-v0")
        class HabitatSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error
