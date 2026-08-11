"""Qwen 프로파일러 실행 진입점.

기존 실행 명령을 유지하기 위한 얇은 래퍼입니다.
실제 구현은 ``qwen_profile`` 패키지에 역할별로 나뉘어 있습니다.
"""

from qwen_profile.runner import run


if __name__ == "__main__":
    run()
