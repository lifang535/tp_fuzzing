"""Native/Extended generation dispatch and shared dimension sampling."""
import random
from typing import List
from src.config import Config, DEFAULT_CONFIG
from src.ir import DataType

class TypeGenerator:
    """Generates shapes from a shared pool — like MLIRSmith's TypeGeneration."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.dim_pool: List[int] = []
        self._init_pool()

    def _init_pool(self):
        self.dim_pool.clear()
        if self.config.easy_shape:
            # Power-of-two sizes; sub-tile shapes still require tail masks.
            for _ in range(self.config.dim_pool_size):
                self.dim_pool.append(random.choice(self.config.easy_shape_values))
        else:
            lo, hi = self.config.dim_range
            for _ in range(self.config.dim_pool_size):
                self.dim_pool.append(random.randint(lo, hi))

    def random_dtype(self) -> DataType:
        choice = random.choice(self.config.supported_dtypes)
        if isinstance(choice, str):
            return DataType(choice)
        return choice


class ProgramGenerator:
    """Mix native regions and the optional typed exploration domain."""
    def __init__(self, config=DEFAULT_CONFIG, backend='tilelang'):
        self.config, self.backend = config, backend
        if not 0 <= config.extended_prob <= 1:
            raise ValueError('extended_prob must be between 0 and 1')
        if config.compile_only and config.extended_prob != 1:
            raise ValueError('compile_only requires extended_prob=1')
        self.type_gen = TypeGenerator(config)
        if config.instance_grid:
            from .grids import GridState
            self.grids = GridState()
        else:
            self.grids = None
        from .region_generator import RegionGenerator
        self.region_gen = RegionGenerator(config, backend, self.type_gen, self.grids)

    def generate(self):
        from src.backends import get_backend
        if self.config.extended_prob and get_backend(self.backend).supports_extended:
            if random.random() < self.config.extended_prob:
                from .extended import ExtendedGenerator
                return ExtendedGenerator(self.config, self.backend,
                                         self.region_gen.feedback, self.grids).generate()
        return self.region_gen.generate()
