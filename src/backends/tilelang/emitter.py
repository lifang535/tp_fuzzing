"""TileLangEmitter entry point for Region and Extended programs."""
from src.backends.common.emitter import ProgramEmitter


class TileLangEmitter(ProgramEmitter):
    def __init__(self, config=None, backend='tilelang'):
        super().__init__(config, backend)
