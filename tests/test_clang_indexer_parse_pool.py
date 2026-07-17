from __future__ import annotations

import os
from pathlib import Path


def test_parse_executor_recycles_long_lived_libclang_workers() -> None:
    from tools.clang_indexer.index_project import make_parse_executor

    with make_parse_executor(max_workers=1, max_tasks_per_child=1) as executor:
        worker_pids = [executor.submit(os.getpid).result(timeout=10) for _ in range(3)]

    assert len(set(worker_pids)) == 3


def test_c_file_with_cpp_construct_keeps_cpp_language_mode(tmp_path: Path) -> None:
    from tools.clang_indexer.index_project import _adapt_args_for_file

    source = tmp_path / "compiler-regression.c"
    source.write_text(
        "char *foo(void) { return(::new char[2] ('a', 'b')); }\n",
        encoding="utf-8",
    )

    adapted = _adapt_args_for_file(["-std=c++17"], str(source))

    assert adapted[:2] == ["-x", "c++"]
    assert "-std=c++17" in adapted

    plain_c = tmp_path / "plain.c"
    plain_c.write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    plain_args = _adapt_args_for_file(["-std=c++17"], str(plain_c))

    assert plain_args[:2] == ["-x", "c"]
    assert "-std=c11" in plain_args


def test_training_document_generation_emits_periodic_heartbeat(capsys) -> None:
    from tools.clang_indexer.index_project import (
        DOCUMENT_HEARTBEAT_ITEMS,
        FunctionDef,
        ProjectIndex,
        build_training_documents,
    )

    index = ProjectIndex()
    total = DOCUMENT_HEARTBEAT_ITEMS + 1
    for item in range(total):
        index.add_function(
            FunctionDef(
                name=f"function_{item}",
                qualified_name=f"function_{item}",
                file=f"file_{item}.cpp",
                line=1,
                text=(
                    f"int function_{item}(int value) {{ int first = value + {item}; "
                    "int second = first * 2; int third = second - value; "
                    "return third + first + second; }"
                ),
                callees=[],
            )
        )

    docs = build_training_documents(index, enriched=False)

    captured = capsys.readouterr()
    assert len(docs) == total
    assert (
        f"Training document generation heartbeat: {DOCUMENT_HEARTBEAT_ITEMS}/{total}"
        in captured.err
    )
