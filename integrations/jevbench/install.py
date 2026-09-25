"""cloneしたJevBenchに、jev_gguf_localアダプタを組み込む(冪等)。

  python integrations/jevbench/install.py ~/jevbench
  JEV_REPO=... JEV_LLAMA_SERVER=... JEV_HEAD=... python -m jevbench.cli run \\
    --adapter jev_gguf_local --endpoint /path/to/jev-Q4_K_M.gguf --tasks datasets/public/easy.jsonl ...
"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
shutil.copyfile(Path(__file__).with_name("jev_gguf_local.py"), root / "jevbench/adapters/jev_gguf_local.py")

init = root / "jevbench/adapters/__init__.py"
text = init.read_text()
line = "from .jev_gguf_local import JevGgufLocalAdapter  # noqa: F401\n"
if line not in text:
    init.write_text(text.rstrip("\n") + "\n" + line)

cli = root / "jevbench/cli.py"
text = cli.read_text()
edits = [
    ("ClassifierDevAdapter, CertoLocalAdapter, QwenFlashLinearAdapter)", "ClassifierDevAdapter, CertoLocalAdapter, QwenFlashLinearAdapter, JevGgufLocalAdapter)"),
    ('"qwen_flash_linear": QwenFlashLinearAdapter}', '"qwen_flash_linear": QwenFlashLinearAdapter,\n             "jev_gguf_local": JevGgufLocalAdapter}'),
    ('"qwen_flash_linear"])', '"qwen_flash_linear", "jev_gguf_local"])'),
]
for old, new in edits:
    if new not in text:
        assert old in text, old
        text = text.replace(old, new)
cli.write_text(text)
print("installed into", root)
