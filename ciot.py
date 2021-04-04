#! /usr/bin/env python3
import argparse
import copy
import enum
import json
import logging
import string
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, TypeVar, Type

PYTHON_REQUIRED = "3.7"
APP_NAME = "ciot"
APP_VERSION = "0.0.1-alpha1"
APP_DESC = """
Command Input Output Testing Tool (ciot)

TBD
"""

LOG = logging.getLogger(APP_NAME)
GLOBAL_TIMEOUT = 10 * 60
FILE_EXTENSIONS = ('in', 'out', 'err', 'files', 'args')

##
# Definitions
##

GeneralDefType = TypeVar('GeneralDefType', bound='GeneralDef')


class GeneralDef:
    """General definition - common base of all definitions"""

    def __init__(self, name: str, desc: str = None):
        self.name: str = name
        self.desc: str = desc if desc else name


class UnitDef(GeneralDef):
    def __init__(self, name: str, desc: str = None, tests: List['TestDef'] = None):
        super().__init__(name, desc)
        self.tests = tests or []


class TestDef(GeneralDef):
    def __init__(self, name: str, desc: str = None,
                 stdin: Path = None, exit_code: Optional[int] = None, args: List[str] = None,
                 env: Dict[str, Any] = None,
                 checks: List['CheckDef'] = None, unit: 'UnitDef' = None):
        super().__init__(name, desc)
        self.stdin: Optional[Path] = stdin
        self.args: List[str] = args
        self.exit_code: Optional[int] = exit_code
        self.checks: Optional[List['CheckDef']] = checks or []
        self.env: Dict[str, Any] = env
        self.unit = unit


class CheckDef(GeneralDef):
    def __init__(self, name: str, desc: str = None, assertion: 'Assertion' = None):
        super().__init__(name, desc)
        self.assertion = assertion


class Assertion:
    def __init__(self, kind: str, params: Dict[str, Any]):
        self.kind = kind
        self.params = params


##
# Parse
##


class UnitFileDefinitionParser:
    def __init__(self, test_dir: Path, data_dir: Path = None):
        self.test_dir = test_dir
        self.data_dir = data_dir

    def parse_unit(self, unit_file: Path) -> 'UnitDef':
        unit_data = load_file(unit_file)
        return self.parse_definition(unit_data, unit_file.stem)

    def parse_definition(self, df: Dict[str, Any], unit_name: str = None) -> UnitDef:
        unit_df = df.get('unit')
        if not unit_df:
            unit_df = {'name': unit_name, 'desc': unit_name}
        if 'name' not in df:
            unit_df['name'] = unit_name

        unit_definition = UnitDef(name=unit_df['name'], desc=unit_df.get('desc'))
        for test_df in df['tests']:
            parsed = self.parse_test_def(unit_df, test_df)
            unit_definition.tests.extend(parsed)

        return unit_definition

    def parse_test_def(self, unit_definition: 'UnitDef', df: Dict[str, Any]) -> List['TestDef']:
        # general params
        name = df['name']
        desc = df.get('desc', name)

        if 'template' in df:
            return self.parse_test_template(unit_definition, df, name, desc)

        stdin = df.get('stdin')
        args = df.get('args', [])
        # Ability to explicitly set to None - null, if null, do not check
        exit_code = df['exit'] if 'exit' in df else 0
        checks = self.parse_checks(df)
        return [TestDef(name, desc, stdin, exit_code, args, checks=checks, unit=unit_definition)]

    def parse_test_template(self, unit_definition: 'UnitDef', df: Dict[str, Any],
                            test_name: str, test_desc: str):
        template = df['template']
        tests = []
        for (idx, case) in enumerate(df['cases']):
            expanded = deep_template_expand(template, case['var'])
            cc = copy.deepcopy(case)
            del cc['var']
            case_df = {**expanded, **cc}
            case_name = case_df.get('name', idx)
            case_desc = case_df.get('desc', idx)
            case_df['name'] = f"{test_name}@{case_name}"
            case_df['desc'] = f"{test_desc} @ {case_desc}"
            tests.extend(self.parse_test_def(unit_definition, case_df))
        return tests

    def parse_checks(self, df: Dict[str, Any]) -> List['CheckDef']:
        checks = []
        stdout = df.get('out', df.get('stdout'))
        if stdout is not None:
            assertion = Assertion(
                FileAssertionRunner.NAME,
                dict(selector="@stdout", expected=self.data_dir / stdout)
            )
            checks.append(CheckDef("stdout_check", "Check the command STDOUT", assertion))

        stderr = df.get('err', df.get('stderr'))
        if stderr is not None:
            assertion = Assertion(
                FileAssertionRunner.NAME,
                dict(selector="@stderr", expected=self.data_dir / stderr)
            )
            checks.append(CheckDef("stderr_check", "Check the command STDERR", assertion))

        exit_code = df.get('exit', df.get('exit_code'))
        if exit_code is not None:
            assertion = Assertion(ExitCodeAssertionRunner.NAME, dict(exit_code=exit_code))
            checks.append(CheckDef("exit_check", "Check the command exit code (main return value)",
                                   assertion))

        files = df.get('files')
        if files is not None and isinstance(files, dict):
            for prov, exp in files.items():
                assertion = Assertion(
                    FileAssertionRunner.NAME,
                    dict(selector=prov, expected=self.data_dir / exp)
                )
                checks.append(CheckDef("file_check", "Check the file content", assertion))

        return checks


##
# Execute
##

class ResultKind(enum.Enum):
    PASS = "pass"
    FAIL = "fail"

    @classmethod
    def check(cls, predicate: bool) -> 'ResultKind':
        return ResultKind.PASS if predicate else ResultKind.FAIL

    def is_pass(self) -> bool:
        return self == self.PASS

    def is_fail(self) -> bool:
        return self == self.FAIL


GeneralResultType = TypeVar('GeneralResultType', bound='GeneralResult')


class GeneralResult:
    @classmethod
    def mk_fail(cls, df: 'GeneralDefType', message: str) -> 'GeneralResultType':
        return cls(df, kind=ResultKind.FAIL, message=message)

    def __init__(self, df: 'GeneralDefType', kind: ResultKind = ResultKind.PASS,
                 message: str = None):
        self.df: 'GeneralDefType' = df
        self.kind = kind
        self.message: str = message
        self.detail: Optional[Dict[str, Any]] = None
        self.sub_results: List['GeneralResultType'] = []

    def add_subresult(self, res: 'GeneralResultType'):
        self.sub_results.append(res)
        if res.kind.is_fail():
            self.kind = res.kind


class UnitRunResult(GeneralResult):
    def __init__(self, df: 'UnitDef'):
        super().__init__(df)


class TestRunResult(GeneralResult):
    def __init__(self, df: 'TestDef'):
        super().__init__(df)


class CheckResult(GeneralResult):
    def __init__(self, df: 'CheckDef', kind: ResultKind, message: str = "",
                 expected=None, provided=None, detail=None, diff=None):
        super().__init__(df, kind=kind, message=message)
        self.expected = expected
        self.provided = provided
        self.diff = diff
        self.detail: Optional[Dict[str, Any]] = detail


class DefinitionRunner:
    def __init__(self, paths: 'Paths'):
        self.paths = paths
        self.assertion_runners = AssertionRunners.instance()

    def run_definition(self, unit_df: UnitDef) -> 'UnitRunResult':
        LOG.info(f"[RUN] Running the suite: {unit_df.name}")
        unit_result = UnitRunResult(unit_df)
        unit_ws = self.paths.unit_workspace(unit_df.name)
        for test_df in unit_df.tests:
            unit_result.add_subresult(self.run_test(test_df, unit_ws))
        return unit_result

    def run_test(self, test_df: 'TestDef', unit_ws: Path) -> 'TestRunResult':
        LOG.info(f"[RUN] Running the test{test_df.name} from {test_df.unit.name}")
        test_result = TestRunResult(test_df)

        try:
            cmd = str(self.paths.binary)
            cmd_res = execute_cmd(cmd,
                                  args=test_df.args,
                                  stdin=test_df.stdin,
                                  nm=test_df.name,
                                  env=test_df.env,
                                  ws=unit_ws)
            ctx = TestCtx(self.paths, test_df, cmd_res)
            for check_df in test_df.checks:
                test_result.add_subresult(self.run_check(ctx, check_df))
            return test_result
        except Exception as e:
            LOG.error("Execution failed: ", e)
            test_result.kind = ResultKind.FAIL
            test_result.message = "Execution failed"
            return test_result

    def run_check(self, ctx: 'TestCtx', check_df: 'CheckDef'):
        LOG.info(f"[RUN] Running Check: {check_df.name} for {ctx.test_df.name}")

        kind = check_df.assertion.kind
        assertion_runner = self.assertion_runners.get(kind)
        if assertion_runner is None:
            return CheckResult.mk_fail(check_df, f"Unable find assertion runner: {kind}")
        instance = assertion_runner(ctx, check_df)
        return instance.evaluate()


class TestCtx:
    def __init__(self, paths: 'Paths', test_df: 'TestDef', cmd_res: 'CommandResult'):
        self.paths = paths
        self.test_df = test_df
        self.cmd_res = cmd_res

    @property
    def ws(self) -> Path:
        return self.paths.unit_workspace(self.test_df.unit.name)


class AssertionRunner:
    NAME = None

    def __init__(self, ctx: 'TestCtx', check_df: 'CheckDef'):
        self.ctx: 'TestCtx' = ctx
        self.check_df: CheckDef = check_df

    @property
    def assertion(self) -> 'Assertion':
        return self.check_df.assertion

    @property
    def params(self) -> Dict['str', Any]:
        return self.assertion.params

    def evaluate(self) -> 'CheckResult':
        return CheckResult.mk_fail(self.check_df, "Unimplemented check")


class FileAssertionRunner(AssertionRunner):
    NAME = "file_cmp"

    def evaluate(self) -> 'CheckResult':
        expected = self.params['expected']
        selector = self.params['selector']
        provided = self._get_provided(selector)
        if not provided.exists():
            return CheckResult.mk_fail(self.check_df, f"Created file does not exists: {selector}")
        return self._compare_files(provided, expected)

    def _get_provided(self, selector: str):
        if selector == "@stdout":
            return self.ctx.cmd_res.stdout
        if selector == "@stderr":
            return self.ctx.cmd_res.stderr

        provided = Path(selector)
        if provided.is_absolute():
            return provided
        return (Path.cwd() / provided).resolve()

    def _compare_files(self, provided: Path, expected: Path):
        nm: str = expected.name

        diff_exec = execute_cmd(
            'diff',
            args=['-u', str(expected), str(provided)],
            ws=self.ctx.ws,
            nm=f"diff-{nm}"
        )
        return CheckResult(
            self.check_df,
            kind=ResultKind.check(diff_exec.exit == 0),
            message="Files content diff",
            expected=expected,
            provided=provided,
            diff=str(diff_exec.stdout),
            detail=diff_exec.as_dict(),
        )


class ExitCodeAssertionRunner(AssertionRunner):
    NAME = "exit_code"

    def evaluate(self) -> 'CheckResult':
        expected = self.params['expected']
        provided = self.ctx.cmd_res.exit
        return CheckResult(
            self.check_df,
            kind=ResultKind.check(provided == expected),
            message="Exit code status",
            provided=provided,
            expected=expected,
            diff="provided != expected"
        )


class AssertionRunners:
    INSTANCE = None

    @classmethod
    def instance(cls) -> 'AssertionRunners':
        if cls.INSTANCE is None:
            cls.INSTANCE = cls.make()
        return cls.INSTANCE

    @classmethod
    def make(cls) -> 'AssertionRunners':
        instance = AssertionRunners()
        instance.add(FileAssertionRunner.NAME, FileAssertionRunner)
        instance.add(ExitCodeAssertionRunner.NAME, ExitCodeAssertionRunner)
        return instance

    def __init__(self):
        self.register: Dict[str, Type[AssertionRunner]] = {}

    def add(self, kind: str, runner: Type[AssertionRunner]):
        self.register[kind] = runner

    def get(self, kind: 'str') -> Optional[Type[AssertionRunner]]:
        return self.register.get(kind)


##
# Utils
##

def load_file(file: Path) -> Any:
    ext = file.suffix
    if ext == '.json':
        with file.open('r') as fd:
            return json.load(fd)
    if ext in ['.yml', '.yaml']:
        try:
            import yaml
        except Exception as ex:
            LOG.error("PyYaml library is not installed")
            raise ex
        with file.open('r') as fd:
            return yaml.safe_load(fd)

    raise Exception(f"Unsupported format: {ext} for {file}")


def deep_template_expand(template: Any, variables: Dict[str, Any]):
    if template is None:
        return None
    if isinstance(template, str):
        return string.Template(template).safe_substitute(variables)
    if isinstance(template, list):
        return [deep_template_expand(i, variables) for i in template]
    if isinstance(template, dict):
        return {k: deep_template_expand(v, variables) for k, v in template.items()}
    return template


def execute_cmd(cmd: str, args: List[str], ws: Path, stdin: Optional[Path] = None,
                stdout: Path = None, stderr: Path = None, nm: str = None,
                log: logging.Logger = None, **kwargs) -> 'CommandResult':
    log = log or LOG
    log.info(f"[CMD]: {cmd} with args {args}")
    nm = nm or cmd
    stdout = stdout or ws / f'{nm}.stdout'
    stderr = stderr or ws / f'{nm}.stderr'
    with stdout.open('w') as fd_out, stderr.open('w') as fd_err:
        fd_in = Path(stdin).open() if stdin else None
        start_time = time.perf_counter_ns()
        exec_result = subprocess.run(
            [cmd, *args],
            stdout=fd_out,
            stderr=fd_err,
            stdin=fd_in,
            **kwargs
        )
        if fd_in:
            fd_in.close()
    end_time = time.perf_counter_ns()
    log.info(f"[CMD] Result: {exec_result}")
    log.debug(f" -> Command stdout {stdout}")
    log.debug(f"STDOUT: {stdout.read_bytes()}")
    log.debug(f" -> Command stderr {stderr}")
    log.debug(f"STDERR: {stderr.read_bytes()}")

    return CommandResult(
        exit_code=exec_result.returncode,
        elapsed=end_time - start_time,
        stdout=stdout,
        stderr=stderr,
    )


class CommandResult:
    def __init__(self, exit_code: int, stdout: Path, stderr: Path, elapsed: int):
        self.exit = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed

    def as_dict(self) -> Dict:
        return {
            'exit': self.exit,
            'stdout': str(self.stdout),
            'stderr': str(self.stderr),
            'elapsed': self.elapsed,
        }


##
# Main CLI
##


class Paths:
    def __init__(self, binary: Path, tests_dir: Path, data_dir: Path = None,
                 artifacts: Path = None):
        self.binary: Path = Path(binary)
        self.tests_dir: Path = Path(tests_dir)
        self.data_dir: Path = Path(data_dir) if data_dir else _resolve_data_dir(tests_dir)
        self.artifacts: Path = Path(artifacts) if artifacts else _make_artifacts_dir()

    def unit_workspace(self, name: str) -> Path:
        ws = self.artifacts / name
        if not ws.exists():
            ws.mkdir(parents=True)
        return ws


def _resolve_data_dir(test_dir: Path) -> Path:
    if (test_dir / 'data').exists():
        return test_dir / 'data'
    return test_dir


def _make_artifacts_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix=APP_NAME))


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(APP_NAME)
    parser.set_defaults(func=None)
    parser.add_argument("-L", "--log-level", type=str,
                        help="Set log level (DEBUG|INFO|WARNING|ERROR)", default='ERROR')
    sub = parser.add_subparsers(title="Sub-Commands")
    # Parse
    sub_parse = sub.add_parser("parse", help="Parse and print the mini hw scenario")
    sub_parse.add_argument('location', type=str, help='Location of the unit file')
    sub_parse.add_argument('-T', '--test-files', type=str, help='Location of the test files',
                           default='tests')
    sub_parse.add_argument('-D', '--test-data-files', type=str,
                           help='Location of the test data files',
                           default=None)
    sub_parse.set_defaults(func=cli_parse)

    # Exec
    return parser


def cli_parse(args):
    unit_file = Path(args.location)
    tests_dir = Path(args.test_files)
    data_dir = args.test_data_files
    paths = Paths(tests_dir=tests_dir, data_dir=data_dir, binary=Path("/usr/bin/echo"))
    parser = UnitFileDefinitionParser(paths.tests_dir, data_dir=paths.data_dir)
    unit_df = parser.parse_unit(unit_file)
    print(dump_json(unit_df))
    return True


def main():
    parser = make_cli_parser()
    args = parser.parse_args()
    _load_logger(args.log_level)
    LOG.debug(f"Parsed args: {args} ")
    if not args.func:
        parser.print_help()
        return
    if not args.func(args):
        print("Execution failed!")


def dump_json(obj, indent=4):
    def _dumper(x):
        if isinstance(x, Path):
            return str(x)
        return x.__dict__

    return json.dumps(obj, default=_dumper, indent=indent)


def _load_logger(level: str = 'INFO'):
    level = level.upper()
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(module)s %(message)s'
            },
            'simple': {
                'format': '%(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },
        },
        'loggers': {
            APP_NAME: {
                'handlers': ['console'],
                'level': level,
            }
        }
    }
    import logging.config
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    main()
