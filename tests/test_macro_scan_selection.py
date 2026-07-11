from __future__ import annotations

from pathlib import Path


def test_macro_scan_roots_are_limited_to_trainable_code_and_indexed_headers(tmp_path: Path) -> None:
    from tools.clang_indexer import index_project

    src = tmp_path / "src"
    include = tmp_path / "include"
    src.mkdir()
    include.mkdir()
    used_cpp = src / "used.cpp"
    unused_cpp = src / "unused.cpp"
    header = include / "api.hpp"
    unused_header = include / "unused.hpp"
    used_cpp.write_text("int used() { return 1; }\n", encoding="utf-8")
    unused_cpp.write_text("// no trainable declarations here\n", encoding="utf-8")
    header.write_text("template <class T> struct Box { T value; };\n", encoding="utf-8")
    unused_header.write_text("#define UNUSED_VALUE 1\n", encoding="utf-8")

    index = index_project.ProjectIndex()
    index.add_function(
        index_project.FunctionDef(
            name="used",
            qualified_name="used",
            file="src/used.cpp",
            line=1,
            text="int used() { return 1; }",
            callees=[],
            is_definition=True,
        )
    )
    index.add_typedef(
        index_project.TypeDef(
            name="Box",
            qualified_name="Box",
            file="include/api.hpp",
            line=1,
            text="template <class T> struct Box { T value; };",
            kind=4,
        )
    )

    roots = index_project.select_macro_scan_files(
        [str(used_cpp), str(unused_cpp), str(header), str(unused_header)],
        index,
        [str(header), str(unused_header)],
        project_dir=str(tmp_path),
    )

    rel_roots = {Path(path).relative_to(tmp_path).as_posix() for path in roots}
    assert rel_roots == {"include/api.hpp", "src/used.cpp"}
