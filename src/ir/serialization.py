"""Persistence for Region, Extended and slice programs and their bug reports."""
from src.ir.region import RegionProgram
from src.ir.extended import ExtendedProgram
from src.ir.slice import SliceProgram


def program_to_dict(program) -> dict:
    if not isinstance(program, (RegionProgram, ExtendedProgram, SliceProgram)):
        raise TypeError(f'Unsupported program: {type(program).__name__}')
    return program.to_dict()


def program_from_dict(data: dict):
    params = data.get('params', {})
    if data.get('type') == 'region':
        return RegionProgram.from_dict(data)
    if data.get('type') == 'extended':
        return ExtendedProgram.from_dict(data)
    if data.get('type') == 'slice':
        return SliceProgram.from_dict(data)
    if 'region_program' in params:
        return RegionProgram.from_dict(params['region_program'])
    if 'extended_program' in params:
        return ExtendedProgram.from_dict(params['extended_program'])
    if 'slice_program' in params:
        return SliceProgram.from_dict(params['slice_program'])
    kind = data.get('type', data.get('compute_kind', 'unknown'))
    raise ValueError(f'Unsupported program format {kind!r}; only Region and Extended programs are supported. Start a new campaign for historical single_op/pipeline/dynamic results.')
