"""Shared completion checks for campaigns, smoke runs and coverage reports.

Compiler processes may die between metadata writes. Incomplete evidence is a
failed attempt, never an exception that aborts the remaining corpus.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class ExtendedEvidence:
    records: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    compiled: bool = False
    diagnostic: str = ''

    def completed(self, mode, stdout):
        marker = 'COMPILE PASSED (no GPU execution)' if mode == 'compile_only' else 'ALL PASSED'
        return (self.compiled and self.progress == {'stage': 'complete', 'variant': mode}
                and marker in stdout.splitlines())


def read_extended_evidence(directory, expected_count):
    directory = Path(directory)
    evidence = ExtendedEvidence()
    try:
        progress = json.loads((directory / 'progress.json').read_text())
        if not isinstance(progress, dict):
            raise ValueError('Invalid progress metadata')
        evidence.progress = progress
    except (OSError, ValueError) as exc:
        evidence.diagnostic = str(exc)

    try:
        records = json.loads((directory / 'compilation.json').read_text())
        if not isinstance(records, list):
            raise ValueError('Invalid compilation manifest')
        labels = set()
        for record in records:
            if (not isinstance(record, dict) or not isinstance(record.get('variant'), str)
                    or not record['variant'] or record['variant'] in labels
                    or type(record.get('complete')) is not bool
                    or not isinstance(record.get('features'), list)
                    or any(not isinstance(f, str) for f in record['features'])
                    or not isinstance(record.get('stages'), dict) or not record['stages']):
                raise ValueError('Invalid or duplicate compilation record')
            labels.add(record['variant'])
            for stage, metadata in record['stages'].items():
                name = record['variant'] + '.' + stage
                if (Path(name).name != name or not isinstance(metadata, dict)
                        or not isinstance(metadata.get('sha256'), str)):
                    raise ValueError('Invalid compiler artifact metadata')
                artifact = directory / name
                if (not artifact.is_file()
                        or hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata['sha256']):
                    raise ValueError('Compiler artifact changed after validation: ' + str(artifact))
        evidence.records = records
        evidence.compiled = (len(records) == expected_count and expected_count > 0
                             and all(r['complete'] for r in records)
                             and evidence.progress.get('stage') in ('reference', 'execute', 'complete'))
        if not evidence.compiled and not evidence.diagnostic:
            evidence.diagnostic = 'Incomplete compilation evidence'
    except (OSError, ValueError) as exc:
        evidence.diagnostic = str(exc)
    return evidence
