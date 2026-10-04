"""Reconstruct the locally tested source; temporary staging helper only."""
from pathlib import Path
import base64
import hashlib
import lzma
import subprocess

BASE='aded8e267f247c84df1948cd9dc2334dce606d91'
EXPECTED_TREE='5ff4d918977f386bfb2a79c0e26a0286628541b8'
PATCH_SHA256='8dfa1eed7a265bb995074f2a999cc49c30edd0898c7001a527afdb1eaea26d39'
subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],check=True)
changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
allowed=lambda p:p.startswith('.implementation/') or p in ('.github/workflows/implementation-source.yml','.github/workflows/em-integration.yml')
assert all(allowed(p) for p in changed),changed
parts=sorted(Path('.implementation').glob('patch*.txt'))
assert len(parts)==4
payload=''.join(p.read_text().strip() for p in parts)
patch=lzma.decompress(base64.b64decode(payload,validate=True))
assert hashlib.sha256(patch).hexdigest()==PATCH_SHA256,'patch transfer checksum mismatch'
Path('/tmp/em-hp.patch').write_bytes(patch)
subprocess.run(['git','apply','--check','/tmp/em-hp.patch'],check=True)
subprocess.run(['git','apply','/tmp/em-hp.patch'],check=True)
subprocess.run(['git','rm','-r','--','.implementation','.github/workflows/implementation-source.yml','.github/workflows/em-integration.yml'],check=True)
subprocess.run(['git','add','-A'],check=True)
tree=subprocess.check_output(['git','write-tree'],text=True).strip()
assert tree==EXPECTED_TREE,(tree,EXPECTED_TREE)
subprocess.run(['git','diff','--cached','--check'],check=True)
print('Exact tested source tree:',tree)
