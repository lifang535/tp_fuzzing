"""TritonEmitter entry point for Region and Extended programs."""
from src.backends.common.emitter import ProgramEmitter


class TritonEmitter(ProgramEmitter):
    def __init__(self, config=None, backend='triton'):
        super().__init__(config, backend)
