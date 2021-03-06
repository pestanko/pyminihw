import abc
import argparse
import shutil
import subprocess
import logging
import os
import tempfile
import time
import enum

from typing import Callable, List, Dict, Set, Optional, Any
from pathlib import Path

APP_NAME = "minihw"
APP_DESC = """
Tool to test the Mini homeworks to PB071
"""

LOG = logging.getLogger(APP_NAME)

EXCLUDED_TASKS = ['template', '.git', 'build']
GLOBAL_TIMEOUT = 10 * 60

CMD_CC = "gcc"
CMD_CFLAGS = ['-Wall', '-Wextra', '-pedantic']
LD_FLAGS = []


class MiniHw:
    def __init__(self, path: Path, name: str = None):
        self.name = name or path.name
        self.path = path
        self._tasks = self._build_tasks(path)

    @property
    def namespace(self) -> List[str]:
        return [self.name]

    @property
    def tasks(self) -> List['Task']:
        if self._tasks is None:
            self._tasks = []

        return self._tasks

    @property
    def build(self) -> Path:
        return self.path / 'build'

    def _build_tasks(self, path=None):
        path = path or self.path
        result = []
        for name in os.listdir(path):
            task_path = path / name
            if not task_path.is_dir() or \
                    name.startswith(".") or \
                    name.startswith("_") or \
                    name in EXCLUDED_TASKS:
                continue
            result.append(Task(self, name, path=task_path))
        return result


class Task:
    def __init__(self, minihw: 'MiniHw', name: str, path: Path = None):
        self.name: str = name
        self.path: Path = path if path else minihw.path / name
        self.minihw: 'MiniHw' = minihw
        self._cases = [
            Case(self, name) for name in find_cases(path)
        ]

    @property
    def namespace(self) -> List[str]:
        return self.minihw.namespace + [self.name]

    @property
    def cases(self) -> List['Case']:
        return self._cases

    @property
    def src_source(self) -> Path:
        return self.path / 'source.c'

    @property
    def src_solution(self) -> Path:
        return self.path / 'solution.c'

    @property
    def build_dir(self) -> Path:
        return self.minihw.path / 'build'

    @property
    def exe_source(self) -> Path:
        return self.build_dir / self.name / f"{self.name}-source"

    @property
    def exe_solution(self) -> Path:
        return self.build_dir / self.name / f"{self.name}-solution"


class Case:
    def __init__(self, task: 'Task', name: str):
        self.task: Task = task
        self.name: str = name
        self.args: List[str] = load_args(task.path, name=name)

    @property
    def namespace(self) -> List[str]:
        return self.task.namespace + [self.name]

    @property
    def stderr(self) -> Path:
        return self._build_path('err')

    @property
    def stdout(self) -> Path:
        return self._build_path('out')

    @property
    def stdin(self) -> Path:
        return self._build_path('in')

    def _build_path(self, ext: str):
        return self.task.path / f"{self.name}.{ext}"


class ResultKind(enum.Enum):
    PASS = 'pass'
    FAIL = 'fail'
    SKIP = 'skip'

    def is_pass(self) -> bool:
        return self == self.PASS

    def is_fail(self) -> bool:
        return self == self.FAIL

    def is_skip(self) -> bool:
        return self == self.SKIP

    def fmt(self, fmt) -> str:
        return fmt(str(self)) if fmt else str(self)

    def __str__(self) -> str:
        return self.value.upper()


class _AnyResult:
    def __init__(self, result: 'ResultKind' = 'pass'):
        self._result: ResultKind = result

    def is_pass(self) -> bool:
        return self.result == ResultKind.PASS

    def is_fail(self) -> bool:
        return self.result == ResultKind.FAIL

    def is_skip(self) -> bool:
        return self.result == ResultKind.SKIP

    @property
    def result(self) -> ResultKind:
        return self._result

    def __str__(self) -> str:
        return self.fmt_str(None)

    def fmt_str(self, fmt=None) -> str:
        return self.res_str(fmt) + " " + self.info_str()

    def res_str(self, fmt: Callable[[str], str] = None) -> str:
        fmt = fmt if fmt else str
        return fmt(f"[{self.result}]")

    def info_str(self) -> str:
        return ""


class MiniHwResult(_AnyResult):
    def __init__(self, minihw: 'MiniHw'):
        super().__init__()
        self.minihw: 'MiniHw' = minihw
        self.tasks: List['TaskResult'] = []

    @property
    def result(self) -> ResultKind:
        for task in self.tasks:
            if task.is_fail():
                return ResultKind.FAIL
        return ResultKind.PASS

    def info_str(self) -> str:
        return f"{self.minihw.name} (tasks: {len(self.tasks)})"


class TaskResult(_AnyResult):
    def __init__(self, task: 'Task'):
        super().__init__()
        self.task: 'Task' = task
        self.cases: List['CaseResult'] = []

    @property
    def result(self) -> ResultKind:
        for case in self.cases:
            if case.is_fail():
                return ResultKind.FAIL
        return ResultKind.PASS

    def info_str(self) -> str:
        return f"{self.task.name} (cases: {len(self.cases)})"


class CaseResult(_AnyResult):
    def __init__(self, case: 'Case', result: ResultKind = ResultKind.PASS, cmd_result=None):
        super().__init__(result=result)
        self.case: 'Case' = case
        self.cmd_result: 'CommandResult' = cmd_result
        self.checks: Dict[str, 'CheckResult'] = {}

    def check(self, checker: 'Checker', msg: str = None):
        res = checker.check(msg)
        LOG.debug(f"Check[{res.is_pass}]: {res}")
        if not res.is_pass:
            self._result = ResultKind.FAIL
        self.checks[checker.name()] = res

    def info_str(self) -> str:
        return f"{self.case.name} (checks: {len(self.checks)})"

    def message(self) -> str:
        res = []
        for check in self.checks.values():
            if check.is_fail:
                res.append(check.fail_msg())
        return "-------------".join(res)


class CheckResult:
    PASS = 'pass'
    FAIL = 'fail'

    def __init__(self, name: str, expected: str, provided: str, message: str, is_pass: bool,
                 diff: str = None, detail: Dict = None):
        self.name = name
        self.expected = expected
        self.provided = provided
        self.message = message
        self.diff = diff
        self.detail = detail
        self.is_pass = is_pass

    @property
    def result(self) -> str:
        return self.__class__.PASS if self.is_pass else self.__class__.FAIL

    @property
    def is_fail(self) -> bool:
        return not self.is_pass

    def __str__(self) -> str:
        s = f"[{self.result.upper()}] {self.name}"
        if self.is_fail:
            s += f" {self.message or ''} provided:{self.provided}; expected:{self.expected}"
        if self.diff:
            s += f" diff:{self.diff}"
        return s

    def fail_msg(self) -> Optional[str]:
        res = f"Result: {self.result}\nProvided: {self.provided}\nExpected: {self.expected}"
        if self.diff:
            res += f'\nDiff: {self.diff}'
            res += f"\nDiff Content: \n{Path(self.diff).read_text()}\n"
        return res


class Checker:
    NAME = None

    @classmethod
    def name(cls) -> str:
        return cls.NAME

    def __init__(self, provided: Any, expected: Any):
        self.expected = expected
        self.provided = provided

    @abc.abstractmethod
    def check(self, msg: str) -> 'CheckResult':
        pass


class PredicateChecker(Checker):
    @abc.abstractmethod
    def predicate(self, provided, expected) -> bool:
        return provided == expected

    def check(self, msg: str) -> 'CheckResult':
        return CheckResult(
            name=self.name(),
            expected=self.expected,
            provided=self.provided,
            is_pass=self.predicate(self.provided, self.expected),
            message=msg,
        )


class ExitValueChecker(PredicateChecker):
    NAME = 'exit'

    def predicate(self, provided, expected) -> bool:
        return provided == expected


class ContentChecker(Checker):
    def __init__(self, provided: Any, expected: Any, ws: Path):
        self.ws = ws
        super().__init__(provided, expected)

    def check(self, msg: str) -> 'CheckResult':
        nm: str = self.expected.name

        diff_exec = execute_cmd(
            'diff',
            args=['-u', str(self.expected), str(self.provided)],
            ws=self.ws,
            nm=f"diff-{nm}"
        )

        return CheckResult(
            name=self.name(),
            expected=self.expected,
            provided=self.provided,
            is_pass=diff_exec.exit == 0,
            message=msg,
            diff=str(diff_exec.stdout),
            detail=diff_exec.as_dict(),
        )


class StdOutChecker(ContentChecker):
    NAME = 'stdout'


class StdErrChecker(ContentChecker):
    NAME = 'stderr'


#######
# Execute
#######

def execute_mini(mini: 'MiniHw', artifacts: Path,
                 mode: str = 'source', task_name: str = None,
                 no_skip: bool = False) -> 'MiniHwResult':
    result = MiniHwResult(mini)
    LOG.info(
        f"[EXEC] minihomework: {mini.name}, artifacts: {artifacts}, mode: {mode}")
    for task in mini.tasks:
        if not task_name or task.name == task_name:
            result.tasks.append(
                execute_task(task, artifacts, mode, no_skip=no_skip)
            )
    return result


def execute_task(task: 'Task', artifacts: Path, mode: str = 'source',
                 no_skip: bool = False) -> 'TaskResult':
    LOG.info(f"[EXEC] task: {task.name}")
    artifacts_task = artifacts / Path(*task.namespace)
    if artifacts_task.exists():
        shutil.rmtree(artifacts_task)
    artifacts_task.mkdir(parents=True)
    result = TaskResult(task)

    for case in task.cases:
        case_result = execute_case(
            case, ws=artifacts_task, mode=mode, no_skip=no_skip)
        _log = LOG.debug if not case_result.is_fail() else LOG.error
        _log(f"Case {case.name} result[{case_result.result}]: {case_result}")
        result.cases.append(case_result)

    return result


def execute_case(case: 'Case', ws: Path, mode: str, no_skip: bool = False) -> 'CaseResult':
    LOG.info(f"[EXEC] case: {case.name}, ws: {ws}")
    exe = case.task.exe_source if mode == 'source' else case.task.exe_solution
    exec_result = _exec_case(exe, case, ws)
    if not exec_result:
        return CaseResult(case, ResultKind.FAIL if no_skip else ResultKind.SKIP)

    result = CaseResult(case, cmd_result=exec_result)
    result.check(ExitValueChecker(exec_result.exit, 0),
                 "Program exited with non-zero value")

    if case.stdout.exists():
        result.check(StdOutChecker(exec_result.stdout, case.stdout, ws=ws))
    if case.stderr.exists():
        result.check(StdErrChecker(exec_result.stderr, case.stderr, ws=ws))
    return result


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


def _exec_case(executable: Path, case: Case, ws: Path) -> Optional[CommandResult]:
    if not executable.exists():
        LOG.warning(f"Executable: {executable} does not exists.")
        return None

    cwd = case.task.path
    result = execute_cmd(
        str(executable),
        args=case.args,
        stdin=case.stdin if case.stdin.exists() else None,
        log=LOG,
        ws=ws,
        cwd=cwd,
        timeout=GLOBAL_TIMEOUT,
    )
    return result

    ######
    # Helpers
    ######


def load_args(path: Path, name: str) -> List[str]:
    file = path / f"{name}.args"
    return file.read_text().splitlines() if file.exists() else []


def find_cases(path: Path) -> Set[str]:
    names = set()
    for file in os.listdir(str(path)):
        full_path = path / file
        if full_path.suffix in ['.out', '.in', '.arg']:
            names.add(full_path.stem)

    return names


def execute_cmd(cmd: str, args: List[str], ws: Path, stdin: Optional[Path] = None,
                stdout: Path = None, stderr: Path = None, nm: str = None,
                log: logging.Logger = None, **kwargs) -> CommandResult:
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


def build_suite(suite, artifacts: Path, clean: bool = False, build: bool = False,
                build_type: str = 'cmake', target: str = None, task_name: str = None) -> bool:
    if suite.build.exists():
        if clean:
            LOG.info(f"[BUILD] Cleaning the build dir: {suite.build}")
            shutil.rmtree(suite.build)
        else:
            LOG.info(
                f"[BUILD] Build dir already exists, no clean mode: {suite.build}")
            return True

    if not suite.build.exists():
        if not build:
            LOG.error(f"[BUILD] Build dir does not exists: {suite.build}")
            return False
        LOG.info(f"[BUILD] Creating the build dir: {suite.build}")
        suite.build.mkdir()
    cwd = suite.build
    if build_type == 'cmake':
        return _build_cmake(artifacts, cwd, suite, target=target, task_name=task_name)
    else:
        return _build_gcc(artifacts, cwd, suite, target=target, task_name=task_name)


def _build_gcc(artifacts: Path, cwd: Path, suite: MiniHw, target: str = None,
               task_name: str = None) -> bool:
    # Execute gcc
    def _build(task, src: Path, out):
        LOG.debug(f"[BUILD] GCC: {src} ~> {out}")
        gcc_res = execute_cmd(
            cmd=CMD_CC, args=[*CMD_CFLAGS, '-o',
                              str(out), str(src), *LD_FLAGS],
            cwd=cwd, ws=artifacts, nm=f"gcc-{out.name}"
        )
        if gcc_res.exit != 0:
            err_print_exec(f"gcc - {task.name} - {src.name}", gcc_res)
            return False
        return True

    for t in suite.tasks:
        if task_name and task_name != t.name:
            continue

        bdir = t.exe_source.parent
        LOG.debug(f"[BUILD] Task: {t.name} (target: {target})")
        if not bdir.exists():
            bdir.mkdir(parents=True)

        if not target or target in ['source', 'both']:
            _build(t, t.src_source, t.exe_source)
        if not target or target in ['solution', 'both']:
            _build(t, t.src_solution, t.exe_solution)

    return True


def _build_cmake(artifacts, cwd, suite, target: str = None, task_name: str = None):
    # Execute cmake
    cmake_res = execute_cmd(
        "cmake", args=[f'-B{cwd}', '-S', str(suite.path)], cwd=cwd, ws=artifacts
    )
    if cmake_res.exit != 0:
        err_print_exec('cmake', cmake_res)
        return False
    # Execute make
    # make_args = extract_make_help(cwd, target, task_name)
    make_res = execute_cmd(
        "make", args=['-k'], cwd=cwd, ws=artifacts
    )
    if make_res.exit != 0:
        err_print_exec('make', make_res)
        return True
    return True


def extract_make_help(cwd, target, task_name):
    make_help = subprocess.run(
        ['make', 'help'],
        cwd=cwd,
        capture_output=True,
    )
    make_args = []
    for line in make_help.stdout.decode(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("..."):
            continue
        line = line[3:].strip()
        if not line.endswith(target):
            continue
        if task_name and not line.startswith(task_name):
            continue
        make_args.append(line)
    LOG.debug(f"Extracted args: {make_args}")
    return make_args


def err_print_exec(name, res):
    print(f"BUILD: Compilation failed - {name} [{res.exit}]"
          f"\nSTDOUT:\n{res.stdout}"
          f"\nSTDERR:\n{res.stderr}"
          f"\nSTDOUT_CONTENT:\n{res.stdout.read_text()}"
          f"\nSTDERR_CONTENT:\n{res.stderr.read_text()}")


def print_suite(suite: 'MiniHw'):
    for task in suite.tasks:
        print(f"Task [{task.name}]")
        for case in task.cases:
            print(f"- {case.name}")
            if case.args:
                print(f"  \tArgs: {case.args}")
            for (nm, pth) in [("in", case.stdin), ("out", case.stdout), ("err", case.stderr)]:
                if pth.exists():
                    print(f"  \t{nm.capitalize()}: {pth}")


class tcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def fail(cls, s: str) -> str:
        return cls._t(cls.FAIL, s)

    @classmethod
    def passed(cls, s: str) -> str:
        return cls._t(cls.OKGREEN, s)

    @classmethod
    def warn(cls, s: str) -> str:
        return cls._t(cls.WARNING, s)

    @classmethod
    def head(cls, s: str) -> str:
        return cls._t(cls.HEADER, s)

    @classmethod
    def _t(cls, c: str, s: str) -> str:
        return f"{c}{s}{cls.ENDC}"


def _result_fmt(res: '_AnyResult', colored: bool = True) -> str:
    if not colored:
        return str(res)

    formatter = tcolors.passed

    if res.is_fail():
        formatter = tcolors.fail
    if res.is_skip():
        formatter = tcolors.warn
    return res.fmt_str(formatter)


def print_suite_result(suite_res: 'MiniHwResult', colored: bool = True):
    def _f(x):
        return _result_fmt(x, colored=colored)

    print(_f(suite_res))
    passed_tasks = 0
    for task in suite_res.tasks:
        print(f"- {_f(task)}")
        if task.is_pass():
            passed_tasks += 1
        for case in task.cases:
            print(f"\t* {_f(case)}")
            for check in case.checks.values():
                if check.is_fail:
                    print(check.fail_msg())

    print(
        f"\n>>> OVERALL RESULT: {_get_result(suite_res, colored)} (passed {passed_tasks} out of {len(suite_res.tasks)} total tasks)")


def _get_result(res: '_AnyResult', colored: bool):
    if not colored:
        return str(res)

    if res.is_fail():
        return tcolors.fail("FAIL")

    if res.is_skip():
        return tcolors.warn("SKIP")

    return tcolors.passed("PASS")


def dump_junit_report(suite_res: 'MiniHwResult', artifacts: Path):
    try:
        import junitparser
    except ImportError:
        LOG.warning("No JUNIT generated - junit parser is not installed")
        return
    reportpath = artifacts / 'junit_report.xml'
    LOG.info(f"[REPORT] Generating JUNIT report: {reportpath}")
    suites = junitparser.JUnitXml()
    for task in suite_res.tasks:
        jsuite = junitparser.TestSuite(name=task.task.name)
        for case in task.cases:
            jcase = junitparser.TestCase(
                name=case.case.name,
                classname='/'.join(case.case.namespace),
                time=case.cmd_result.elapsed / 1000000.0 if case.cmd_result else 0
            )
            if case.is_fail():
                jcase.result = [junitparser.Failure(
                    c.fail_msg()) for c in case.checks.values() if c.is_fail]
                if case.cmd_result:
                    jcase.system_out = str(case.cmd_result.stdout)
                    jcase.system_err = str(case.cmd_result.stderr)
            elif case.is_skip():
                jcase.result = [junitparser.Skipped()]
            jsuite.add_testcase(jcase)
        suites.add_testsuite(jsuite)
    suites.write(str(reportpath))


def _load_logger(level: str = 'INFO'):
    level = level.upper()
    LOGGING = {
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
    logging.config.dictConfig(LOGGING)


###########
# MAIN
###########


def parse_cli_args():
    parser = argparse.ArgumentParser(APP_NAME, description=APP_DESC)
    parser.set_defaults(func=None)
    parser.add_argument("-L", "--log-level", type=str,
                        help="Set log level (DEBUG|INFO|WARNING|ERROR)", default='ERROR')
    sub = parser.add_subparsers(title="Sub-Commands")
    # parse
    sub_parse = sub.add_parser(
        "parse", help="Parse and print the mini hw scenario")
    sub_parse.add_argument('location', type=str,
                           help='Location of the mini homework', default='.')
    sub_parse.set_defaults(func=cli_parse)
    # build
    sub_build = sub.add_parser("build", help="Build the minihomework")
    sub_build.add_argument('location', type=str,
                           help='Location of the mini homework', default='.')
    sub_build.add_argument('-C', "--clean", action="store_true",
                           help="Clean the build dir", default=False)
    sub_build.add_argument("-A", "--artifacts", type=str,
                           help="Artifacts directory", default=None)
    sub_build.add_argument("-T", "--target", type=str,
                           help="Target to build (source|solution|both)", default=None)
    sub_build.add_argument('--build-type', type=str,
                           help="Build type (cmake|gcc)", default='cmake')
    sub_build.set_defaults(func=cli_build)
    # execute
    sub_exe = sub.add_parser("execute", help="Execute the minihomework")
    sub_exe.add_argument('location', type=str,
                         help='Location of the mini homework', default='.')
    sub_exe.add_argument('-C', "--clean", action="store_true",
                         help="Clean the build dir", default=False)
    sub_exe.add_argument('-B', "--build", action="store_true",
                         help="Build the solution if not built", default=False)
    sub_exe.add_argument("-A", "--artifacts", type=str,
                         help="Artifacts directory", default=None)
    sub_exe.add_argument("-T", "--target", type=str,
                         help="Target to exec (source|solution)", default='source')
    sub_exe.add_argument("-t", "--task", type=str,
                         help="Select task to be executed", default=None)
    sub_exe.add_argument('--build-type', type=str,
                         help="Build type (cmake|gcc)", default='cmake')
    sub_exe.add_argument('--no-color', action='store_true',
                         help="Disable colored output", default=False)
    sub_exe.set_defaults(func=cli_exec)
    return parser


def cli_parse(args):
    path = Path(args.location).resolve()
    LOG.info(f"Parsing the {APP_NAME} ({path.name}) at location {path}")
    suite = MiniHw(path)
    print_suite(suite)
    return True


def cli_build(args):
    path = Path(args.location).resolve()
    artifacts = get_artifacts(args.artifacts)
    LOG.info(
        f"Building the {APP_NAME} ({path.name}) at location {path}; artifacts: {artifacts}")
    suite = MiniHw(path)
    return build_suite(suite, clean=args.clean, build=True, artifacts=artifacts,
                       build_type=args.build_type)


def cli_exec(args):
    path = Path(args.location).resolve()
    artifacts = get_artifacts(args.artifacts)
    LOG.info(
        f"Executing the {APP_NAME} ({path.name}) at location {path}; artifacts: {artifacts}")
    suite = MiniHw(path)
    if not build_suite(
            suite,
            artifacts=artifacts,
            clean=args.clean,
            build=args.build,
            build_type=args.build_type,
            target=args.target,
            task_name=args.task):
        return False
    suite_res = execute_mini(
        suite,
        artifacts=artifacts,
        mode=args.target,
        task_name=args.task,
        no_skip=True,
    )
    print()
    print_suite_result(suite_res, colored=not args.no_color)
    dump_junit_report(suite_res, artifacts)
    return suite_res.is_pass()


def main():
    parser = parse_cli_args()
    args = parser.parse_args()
    _load_logger(args.log_level)
    LOG.debug(f"Parsed args: {args} ")
    if not args.func:
        parser.print_help()
        return 0
    if not args.func(args):
        print("Execution failed!")
        return 1

    return 0


def get_artifacts(value: str, name: str = None) -> Path:
    if not value:
        return Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-{name}"))
    return ((Path(value) / name) if name else Path(value)).resolve()


if __name__ == "__main__":
    import sys
    sys.exit(main())
