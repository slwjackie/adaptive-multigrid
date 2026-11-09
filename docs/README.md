# neural-amg 문서

이 브랜치의 이산화는 **비정렬 삼각형 P1 FEM만** 제공한다.

| 문서 | 내용 |
| --- | --- |
| [한국어 README](../README_KR.md) | 현재 구현과 실행 흐름 |
| [영문 README](../README.md) | 수치 구성과 코드 위치 |
| [실행 문서](neural_amg.md) | 설치·학습·재개·freeze·final/OOD·입력 특징 |
| [P1 수식과 검증](p1_fem.md) | 삼각형 격자, FEM 조립, 하중·경계, 오차 수렴 |
| [Smoke 설정](../configs/neural_amg_smoke.json) | 작은 전체 학습·평가 실행 |
| [Research 설정](../configs/neural_amg_research.json) | 560 TRAIN / 280 validation / 280 final 및 OOD |

이전 Q1·정렬 P1·CNN·EM-line 실험은 이 브랜치에서 실행하지 않는다.
기존 문서는 [출발 커밋의 문서 목록](https://github.com/slwjackie/adaptive-multigrid/blob/8892f31520062dc832d33ae47c58216185ce5f11/docs/README.md)에 보존되어 있다.
