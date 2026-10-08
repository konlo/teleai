"""Launch the production persistent agent using its pinned environment."""
import argparse
from pathlib import Path
import os
import socket
import sys


def port_in_use(port):
    for family, address in ((socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.2)
                if connection.connect_ex((address, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8502, help='Local Streamlit port (default: 8502)')
    args, streamlit_args = parser.parse_known_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    root = Path(__file__).resolve().parents[1]
    python = root / '.telly_runtime/v1-venv/bin/python'
    if not python.exists():
        raise SystemExit('Install requirements.txt in .telly_runtime/v1-venv first.')
    if port_in_use(args.port):
        raise SystemExit(
            f'{args.port} 포트에서 이미 서버가 실행 중입니다. 기존 프로세스를 종료한 뒤 Telly를 다시 실행하세요.'
        )
    os.chdir(root)
    os.execv(str(python), [str(python), '-m', 'streamlit', 'run', 'main.py',
                           '--server.address=127.0.0.1', f'--server.port={args.port}',
                           *streamlit_args])


if __name__ == '__main__':
    main(sys.argv[1:])
