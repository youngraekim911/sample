# -*- coding: utf-8 -*-
"""단독 실행 파일(.exe/앱) 빌드 — 윈도우(또는 mac/linux) PC 에서 1회만 실행.

받는 사람이 파이썬조차 없어도 되는 진짜 단독 실행본을 만든다. 파이썬+torch 등
모든 의존성을 통째로 번들.

준비:
    python -m pip install -r requirements.txt      # 실행 의존성
    python -m pip install pyinstaller               # 빌드 도구

빌드:
    python build_exe.py

결과:
    dist/cis_rcwa/            <- 이 폴더 전체를 zip 해서 배포
        cis_rcwa(.exe)       <- 받는 사람은 이걸 더블클릭 (파이썬 불필요)
        data/ conf/ ...      <- 물질 n,k 는 여기 data/materials/*.txt 를 교체

주의:
- GPU(CUDA)로 쓰려면 이 빌드 PC 에 CUDA 버전 torch 를 설치한 상태로 빌드.
  (CPU torch 로 빌드하면 CPU 전용 실행본이 됨 — 그래도 동작)
- onedir(폴더) 방식 권장. torch 는 onefile(단일 exe)이면 시작이 느리고
  간혹 실패하므로 폴더째 배포가 안전.
- 첫 실행 시 exe 옆에 data/·conf/ 가 자동 생성되고, 물질/구조를 거기서 편집.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = ";" if os.name == "nt" else ":"


def main():
    try:
        import PyInstaller.__main__ as pim
    except ImportError:
        print("PyInstaller 가 없습니다.  python -m pip install pyinstaller  후 다시 실행.")
        sys.exit(1)

    args = [
        os.path.join(HERE, "app.py"),
        "--name", "cis_rcwa",
        "--noconfirm", "--clean",
        "--console",                       # 상태/오류 표시 (원하면 --windowed 로 숨김)
        # 읽기전용 자원 번들 (exe 안). 실행 시 exe 옆으로 data/conf 를 seed.
        "--add-data", f"{os.path.join(HERE, 'editors')}{SEP}editors",
        "--add-data", f"{os.path.join(HERE, 'data')}{SEP}data",
        "--add-data", f"{os.path.join(HERE, 'conf')}{SEP}conf",
        # torch/matplotlib 은 데이터·동적라이브러리가 많아 전량 수집 필수
        "--collect-all", "torch",
        "--collect-all", "matplotlib",
        "--collect-submodules", "src",
        # 시작 시간/용량 절감: 무거운 미사용 백엔드 제외 (있으면)
        "--exclude-module", "tkinter",
    ]
    print("[build] PyInstaller 실행 (수 분 소요, torch 수집이 오래 걸립니다)...")
    pim.run(args)
    out = os.path.join(HERE, "dist", "cis_rcwa")
    print("\n[build] 완료 ->", out)
    print("[build] 이 폴더(dist/cis_rcwa) 전체를 zip 해서 배포하세요.")
    print("[build] 받는 사람: 폴더 안 cis_rcwa" + (".exe" if os.name == "nt" else "") +
          " 더블클릭 (파이썬 불필요).")


if __name__ == "__main__":
    main()
