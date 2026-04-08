from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_start_script_reads_existing_docs_directory_and_stop_keeps_it(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "nested" / "runtime"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_file = docs_dir / "custom.txt"
    source_file.write_text("Semantic embeddings improve retrieval quality.\n", encoding="utf-8")
    start_script = REPO_ROOT / "scripts" / "start.sh"
    stop_script = REPO_ROOT / "scripts" / "stop.sh"

    start = subprocess.run(
        [
            "bash",
            str(start_script),
            "--runtime-dir",
            str(runtime_dir),
            "--docs-dir",
            str(docs_dir),
            "--skip-sync",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert start.returncode == 0, start.stderr
    assert (runtime_dir / "settings.yaml").exists()
    assert (runtime_dir / "store.json").exists()
    assert (runtime_dir / "trace.jsonl").exists()
    assert (runtime_dir / "session.env").exists()
    assert not (docs_dir / "getting-started.txt").exists()
    assert str(source_file) in start.stdout
    assert "Previewed 1 document(s)." in start.stdout
    assert "src.interfaces.cli.chat" in start.stdout
    assert "Demo environment ready" in start.stdout

    stop = subprocess.run(
        [
            "bash",
            str(stop_script),
            "--runtime-dir",
            str(runtime_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert stop.returncode == 0, stop.stderr
    assert not runtime_dir.exists()
    assert docs_dir.exists()
    assert source_file.exists()


@pytest.mark.e2e
def test_start_script_creates_missing_docs_directory(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "nested" / "runtime"
    docs_dir = tmp_path / "missing-docs"
    start_script = REPO_ROOT / "scripts" / "start.sh"
    stop_script = REPO_ROOT / "scripts" / "stop.sh"

    start = subprocess.run(
        [
            "bash",
            str(start_script),
            "--runtime-dir",
            str(runtime_dir),
            "--docs-dir",
            str(docs_dir),
            "--skip-sync",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert start.returncode == 0, start.stderr
    assert docs_dir.exists()
    assert (docs_dir / "getting-started.txt").exists()
    assert (runtime_dir / "store.json").exists()
    assert "Created docs directory" in start.stdout
    assert "Previewed 1 document(s)." in start.stdout

    stop = subprocess.run(
        [
            "bash",
            str(stop_script),
            "--runtime-dir",
            str(runtime_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert stop.returncode == 0, stop.stderr
    assert not runtime_dir.exists()
    assert docs_dir.exists()
    assert (docs_dir / "getting-started.txt").exists()


@pytest.mark.e2e
def test_start_script_switches_to_pdf_mode_for_pdf_documents(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "nested" / "runtime"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    runtime_dir.mkdir(parents=True)
    stale_config = runtime_dir / "settings.yaml"
    stale_config.write_text(
        'adapters:\n  loader:\n    provider: "text"\ningestion:\n  supported_extensions:\n    - ".txt"\n',
        encoding="utf-8",
    )
    pdf_fixture = REPO_ROOT / "tests" / "fixtures" / "ingestion" / "simple.pdf"
    pdf_file = docs_dir / "simple.pdf"
    shutil.copyfile(pdf_fixture, pdf_file)
    start_script = REPO_ROOT / "scripts" / "start.sh"
    stop_script = REPO_ROOT / "scripts" / "stop.sh"

    start = subprocess.run(
        [
            "bash",
            str(start_script),
            "--runtime-dir",
            str(runtime_dir),
            "--docs-dir",
            str(docs_dir),
            "--skip-sync",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert start.returncode == 0, start.stderr
    assert 'loader="pdf"' not in start.stdout
    assert "loader=pdf" in start.stdout
    assert "Previewed 1 document(s)." in start.stdout
    config_text = (runtime_dir / "settings.yaml").read_text(encoding="utf-8")
    assert 'provider: "pdf"' in config_text
    assert '- ".pdf"' in config_text
    assert "dense_candidate_multiplier: 3" in config_text
    assert "candidate_results: 6" in config_text

    documents = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.interfaces.cli.documents",
            "list",
            "--collection",
            "knowledge",
            "--config",
            str(runtime_dir / "settings.yaml"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert documents.returncode == 0, documents.stderr
    assert "Found 1 document(s)." in documents.stdout
    assert str(pdf_file) in documents.stdout

    stop = subprocess.run(
        [
            "bash",
            str(stop_script),
            "--runtime-dir",
            str(runtime_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert stop.returncode == 0, stop.stderr
    assert not runtime_dir.exists()
    assert docs_dir.exists()
    assert pdf_file.exists()


@pytest.mark.e2e
def test_stop_script_refuses_unmanaged_directory(tmp_path: Path) -> None:
    unmanaged_dir = tmp_path / "unsafe-runtime"
    unmanaged_dir.mkdir()
    stop_script = REPO_ROOT / "scripts" / "stop.sh"

    stop = subprocess.run(
        [
            "bash",
            str(stop_script),
            "--runtime-dir",
            str(unmanaged_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert stop.returncode == 1
    assert "not created by scripts/start.sh" in stop.stderr
    assert unmanaged_dir.exists()
