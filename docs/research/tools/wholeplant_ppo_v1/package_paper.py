"""Package editable LaTeX and verify compilation after archive extraction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


FILES = ("plantanagent-wholeplant-ppo.tex", "interpretation.tex", "results.tex", "findings.tex",
         "README.md", "REPORT.md", "report-data.json", "figures/contribution-comparison.pdf",
         "figures/contribution-comparison.png", "figures/training-curves.pdf", "figures/training-curves.png", "plantanagent-wholeplant-ppo.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writeup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.qa.exists():
        raise FileExistsError("Keep prior package attempts; use fresh output and QA paths")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "files": [],
                "build": "pdflatex -no-shell-escape plantanagent-wholeplant-ppo.tex (twice)",
                "source_date_epoch": 1789603200}
    for name in FILES:
        source = args.writeup / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing regular source artifact: {source}")
        content = source.read_bytes()
        manifest["files"].append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    args.qa.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plantanagent-paper-package-") as directory:
        temporary = Path(directory)
        archive_path = temporary / "source.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in manifest["files"]:
                archive.write(args.writeup / item["path"], item["path"])
            archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        extracted = temporary / "extracted"
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Source archive failed CRC verification")
            archive.extractall(extracted)
        for item in manifest["files"]:
            if hashlib.sha256((extracted / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Extracted source hash differs")
        build = extracted / ".build"
        build.mkdir()
        env = {**os.environ, "SOURCE_DATE_EPOCH": "1789603200", "FORCE_SOURCE_DATE": "1", "TZ": "UTC"}
        for _ in range(2):
            command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                       "-output-directory=.build", "plantanagent-wholeplant-ppo.tex"]
            completed = subprocess.run(command, cwd=extracted, env=env, capture_output=True, text=True, timeout=45)
            (args.qa / "compile-output.txt").write_text(completed.stdout + completed.stderr)
            completed.check_returncode()
        log = (build / "plantanagent-wholeplant-ppo.log").read_text()
        (args.qa / "standalone.log").write_text(log)
        pages = re.search(r"Output written on .*?\((\d+) pages?[,\s]", log.replace("\n", ""))
        if not pages or int(pages.group(1)) != 2 or any(issue in log for issue in ("Overfull", "undefined", "multiply defined")):
            raise ValueError("Extracted source did not produce two clean pages")
        texts = []
        for pdf in (args.writeup / "plantanagent-wholeplant-ppo.pdf", build / "plantanagent-wholeplant-ppo.pdf"):
            text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True, timeout=15)
            texts.append(" ".join(text.split()))
        if texts[0] != texts[1]:
            raise ValueError("Standalone PDF text differs from the reviewed report")
        shutil.copyfile(build / "plantanagent-wholeplant-ppo.pdf", args.qa / "standalone.pdf")
        with args.output.open("xb") as target, archive_path.open("rb") as source:
            shutil.copyfileobj(source, target)
    receipt = {"status": "passed", "archive": str(args.output.resolve()), "bytes": args.output.stat().st_size,
               "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "files": len(FILES),
               "pages": 2, "standalone_text_matches": True, "manifest": manifest}
    (args.qa / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
