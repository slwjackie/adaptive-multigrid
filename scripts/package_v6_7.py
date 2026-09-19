#!/usr/bin/env python3
"""Build a source/result ZIP, fresh manifests and a SHA-256 sidecar."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def include(path):
    rel=path.relative_to(ROOT)
    if any(p in {'__pycache__','.pytest_cache','.git','build','dist'} or p.endswith('.egg-info') for p in rel.parts):
        return False
    return path.is_file() and path.suffix not in {'.pyc','.so','.dylib','.pyd','.tmp'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT.parent/'adaptive_neural_multigrid_v6_7_final.zip'))
    args=p.parse_args();destination=Path(args.output).resolve()
    if destination.is_relative_to(ROOT):raise ValueError('Write the archive outside the project root')
    files=sorted(p for p in ROOT.rglob('*') if include(p))
    sources=[f for f in files if f.relative_to(ROOT).parts[0] in {'src','scripts','tests','configs'} or f.name in {'pyproject.toml','requirements.txt'}]
    (ROOT/'SOURCE_MANIFEST.sha256').write_text(''.join(f'{digest(f)}  {f.relative_to(ROOT).as_posix()}\n' for f in sources))
    files=sorted(p for p in ROOT.rglob('*') if include(p) and p.name!='PACKAGE_MANIFEST.sha256')
    (ROOT/'PACKAGE_MANIFEST.sha256').write_text(''.join(f'{digest(f)}  {f.relative_to(ROOT).as_posix()}\n' for f in files))
    files=sorted(p for p in ROOT.rglob('*') if include(p))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,f'{ROOT.name}/{f.relative_to(ROOT).as_posix()}')
    sha=digest(destination)
    destination.with_name(destination.name+'.sha256').write_text(f'{sha}  {destination.name}\n')
    print(f'Archive: {destination}\nFiles: {len(files)}\nSource manifest: {len(sources)}\nSHA256: {sha}')
if __name__=='__main__':main()
