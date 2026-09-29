"""Check publishable files for credentials and deployment identifiers.

Print file/line/rule only. This check supplements a credential scanner; it
does not inspect older remote branches or generated local output folders.
"""
import ipaddress
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "outputs", "__pycache__", ".pytest_cache"}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})\b"),
    "home-path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    "credential-url": re.compile(r"https?://[^\s/'\"]+:[^\s/'\"]+@"),
    "literal-credential": re.compile(r"(?i)(?:password|passwd|api_key|access_token|auth_token)\s*[:=]\s*['\"][A-Za-z0-9_+/.=-]{8,}['\"]"),
}
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")


def files(root):
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if any(part in EXCLUDED or part.startswith(".venv-") for part in parts):
            continue
        if path.is_file():
            yield path


def scan(root=ROOT):
    findings = []
    count = 0
    for path in files(root):
        count += 1
        relative = str(path.relative_to(root))
        if path.is_symlink():
            findings.append((relative, 0, "symlink"))
            continue
        if path.suffix in {".pyc", ".pem", ".key", ".log"} or path.name.startswith(("id_rsa", "id_ed25519")):
            findings.append((relative, 0, "private-or-generated-file"))
        if path.stat().st_size > 45 * 1024 * 1024:
            findings.append((relative, 0, "oversize-file"))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append((relative, 0, "unexpected-binary"))
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((relative, number, label))
            for match in IPV4.finditer(line):
                try:
                    address = ipaddress.ip_address(match.group())
                except ValueError:
                    continue
                if not address.is_loopback and not address.is_unspecified:
                    findings.append((relative, number, "literal-deployment-ip"))
    return count, findings


if __name__ == "__main__":
    count, findings = scan()
    for path, line, rule in findings:
        print("%s:%d: %s" % (path, line, rule))
    print("Scanned %d files; %d findings" % (count, len(findings)))
    sys.exit(bool(findings))
