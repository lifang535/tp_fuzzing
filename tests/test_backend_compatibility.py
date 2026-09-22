"""Execute saved current IR through both backends after removal of obsolete routes.

Fixtures were captured before cleanup. Compare complete function ASTs, excluding
removed unused helpers and module entry printing; do not regenerate on failure.
"""
import ast
import hashlib
import json
from pathlib import Path
import unittest
from src.config import Config
from src.ir.serialization import program_from_dict, program_to_dict
from src.workflow.oracle import Oracle
from src.workflow.fuzzer.fuzzer import TileSmith


def emission_digest(code):
    nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()).hexdigest()


class BackendCompatibilityTests(unittest.TestCase):
    def test_saved_current_programs_preserve_emitted_behavior(self):
        records = json.loads(Path(__file__).with_name('fixtures').joinpath('current_programs.json').read_text())
        for record in records:
            with self.subTest(case=record['case']):
                program = program_from_dict(record['program'])
                code = Oracle(Config(), record['backend'])._emit_code(program)
                self.assertEqual(emission_digest(code), record['emission'])
                restored = program_from_dict(program_to_dict(program))
                self.assertEqual(TileSmith._make_sig(program), TileSmith._make_sig(restored))
                self.assertEqual(TileSmith._make_sig(program), TileSmith._make_sig_from_dict(record['program']))

    def test_removed_formats_are_rejected(self):
        for kind in ('single_op', 'pipeline', 'dynamic'):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Unsupported program format'):
                program_from_dict({'type': kind, 'params': {}})
        record = json.loads(Path(__file__).with_name('fixtures').joinpath('current_programs.json').read_text())[0]
        payload = dict(record['program'], legacy={'type': 'single_op'})
        with self.assertRaisesRegex(ValueError, 'Legacy region wrappers'):
            program_from_dict(payload)
        with self.assertRaises(TypeError):
            Oracle(Config())._emit_code(object())


if __name__ == '__main__':
    unittest.main()
