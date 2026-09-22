"""Bounded compiler execution, including compiler subprocesses on timeout."""
import os
import signal
import subprocess


def run_isolated(command, timeout, **options):
    options.setdefault('start_new_session', os.name == 'posix')
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, **options) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if os.name == 'posix' and options['start_new_session']:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
