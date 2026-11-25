# 🚀 Git & GitHub 핵심 정리

> **강의 참고:** git + GitHub - On your mark, git set, go (강사 최인규) & 주요 CLI 명령어 정리

## 📑 목차

1. [Git과 버전 관리](#1-git과-버전-관리)
2. [리눅스(CLI) 필수 명령어](#2-리눅스cli-필수-명령어)
3. [Git & GitHub 기초 (CLI)](#3-git--github-기초-cli)
4. [Git & GitHub 심화 (Branch, Conflict, PR)](#4-git--github-심화-branch-conflict-pr)
5. [실전 협업을 위한 팁 (.gitignore, Markdown, Convention)](#5-실전-협업을-위한-팁)

---

## 1. Git과 버전 관리

### 💡 버전 관리 시스템(VCS)이 필요한 이유

* 게임의 **SAVE(저장) 기능**과 같아서 원하는 시점으로 언제든 되돌아갈 수 있습니다.
* `최종`, `최최종`, `진짜최종`, `컨펌_v1` 같은 무분별한 파일/폴더 복사를 방지합니다.
* **핵심 이점:** 변경점 추적(누가, 언제, 무엇을 수정했는지 기록), 백업 및 복구, 원활한 팀 협업(코드 병합).

### 🔍 Git이란?

* **분산 버전 관리 시스템(DVCS):** 컴퓨터 파일의 변경사항을 추적하고 여러 사용자 간의 작업을 조율하는 시스템입니다. (리누스 토르발스 개발)
* 원하는 시점마다 깃발(`Commit`)을 꽂고 자유롭게 이동할 수 있습니다.
* **CLI**(Command Line Interface)와 **GUI**(Graphic User Interface) 방식을 모두 지원합니다.

### ⚙️ 초기 환경 설정

1. **프로그램 설치:** [Git 공식 사이트](https://git-scm.com/) 및 [VS Code](https://code.visualstudio.com/) 설치
2. **GitHub 가입:** [GitHub](https://github.com/) 계정 생성
3. **사용자 정보 등록 (최초 1회 필수):**

   ```bash
   git config --global user.name "사용자이름"
   git config --global user.email "이메일주소"
   
   # 설정 확인
   git config --global --list
   ```

> **💡 참고 (Windows 인증 오류 시):** 제어판의 **자격 증명 관리자 > Windows 자격 증명**에서 `git:https://github.com` 항목의 사용자 이름과 암호(Personal Access Token)를 수정할 수 있습니다.

---

## 2. 리눅스(CLI) 필수 명령어

### 🧭 경로 및 홈 디렉토리 기본 개념

리눅스/Unix 계열 시스템에서 모든 경로는 최상위인 `/` (root)에서 시작합니다.

| 키워드 | 설명 |
| :---: | :--- |
| `/` | 모든 경로의 시작점 (루트 디렉토리) |
| `~` | 현재 사용자의 홈 디렉토리 (Home) |
| `.` | 현재 작업 중인 디렉토리 |
| `..` | 현재 디렉토리의 상위(부모) 폴더 |

* **운영체제별 홈 디렉토리(`~`) 위치:**
  * **macOS / Linux:** `/home/사용자명` (macOS는 `/Users/사용자명`)
  * **Windows (Git Bash):** `/c/Users/사용자명`

### 📂 파일/폴더 위치 및 탐색 명령어

```bash
# 1. 현재 내 위치(경로) 확인 - 길을 잃었을 때 가장 먼저 확인!
pwd

# 2. 현재 폴더 내 파일 및 폴더 목록 확인
ls          # 기본 목록 조회
ls -l       # 권한 / 소유자 / 수정시간까지 자세히 조회
ls -a       # 숨김 파일(.git 등)까지 모두 조회
ls -al      # 숨김 파일 포함 상세 조회 (가장 많이 사용되는 조합)

# 3. 폴더 이동하기 (TIP: Tab 키를 누르면 폴더명 자동완성)
cd 프로젝트폴더명   # 해당 폴더로 이동
cd ..             # 상위 폴더로 이동
cd ~              # 홈 디렉토리로 돌아가기

# 4. 현재 폴더를 Windows 탐색기로 열기
start .
```

> **💡 `ls -al`이 자주 쓰이는 이유:** `.git` 숨김 폴더가 존재하는지 확인하여 Git이 초기화된 폴더인지 체크하거나, 파일 접근 권한 문제를 확인할 때 유용합니다.

### 🛠️ 파일/폴더 조작 명령어

```bash
# 1. 새 폴더 만들기
mkdir myapp
mkdir -p src/components        # 중첩된 하위 폴더까지 한 번에 생성

# 2. 빈 파일 만들기
touch index.html
touch app.js README.md         # 여러 파일 동시 생성 가능

# 3. 파일 및 폴더 삭제 (주의: 휴지통을 거치지 않고 즉시 삭제됨)
rm test.txt                    # 파일 삭제
rm -r myapp                    # 폴더 삭제
rm -rf myapp                   # 폴더 강제 삭제

# 4. 이동 또는 이름 변경
mv a.txt b.txt                 # a.txt의 이름을 b.txt로 변경
mv a.txt backup/               # a.txt를 backup 폴더로 이동

# 5. 파일 및 폴더 복사
cp a.txt backup.txt            # 파일 복사
cp -r src backup               # 폴더 전체 복사
```

### 📄 파일 내용 확인 및 검색

| 명령어 | 역할 | 대표 사용 사례 |
| :--- | :--- | :--- |
| `cat` | 파일 내용을 한 번에 전부 출력 | 짧은 파일 내용 확인 |
| `head` | 파일의 위쪽 일부만 출력 | 설정 파일 구조 훑어보기 |
| `tail` | 파일의 아래쪽 일부만 출력 | 최근 로그 확인 |
| `tail -f` | 파일 끝부분을 실시간으로 추적 출력 | 서버 실행 로그(`log`) 실시간 모니터링 |
| `grep` | 파일 내 특정 문자열 검색 | 에러 로그나 특정 환경변수 찾기 |

```bash
cat README.md                      # 파일 전체 출력
echo "내용 추가" >> README.md      # 파일 끝에 텍스트 추가

head -n 10 config.json             # 처음 10줄만 보기
tail -n 20 logs/app.log            # 마지막 20줄만 보기
tail -f logs/app.log               # 실시간 로그 모니터링 (종료: Ctrl + C)

grep "DB_HOST" .env                # .env 파일에서 "DB_HOST" 검색
grep -n "ERROR" logs/app.log       # 줄 번호(-n)와 함께 "ERROR" 검색
```

### 🖥️ 시스템 정보 및 기타 참고 명령어

| 명령어 | 설명 |
| :--- | :--- |
| `whoami` | 현재 로그인한 사용자 계정 확인 |
| `which <명령어>` | 명령어(실행 파일)의 실제 설치 위치 확인 |
| `du -sh` | 현재 폴더나 특정 파일의 용량 확인 |
| `df -h` | 시스템 전체 디스크 사용량 확인 |
| `chmod` | 파일/폴더의 실행 및 접근 권한 변경 |
| `chown` | 파일/폴더의 소유자 변경 (관리자 권한 필요) |
| `tar` | 파일 압축 및 압축 해제 |
| `curl` | URL의 내용을 가져와 출력하거나 저장 |
| `wget` | URL에서 파일을 직접 다운로드 |

---

## 3. Git & GitHub 기초 (CLI)

### 📂 Git의 3가지 작업 영역

Git은 파일의 상태를 아래 3단계 영역으로 나누어 관리합니다.

1. **Working Directory (작업 디렉토리):** 실제 코드 작업을 수행하는 로컬 폴더
2. **Staging Area (스테이징 영역):** 커밋(저장)할 후보 파일들을 임시로 모아두는 공간
3. **Repository (저장소):** 버전 이력(`Commit`)이 영구적으로 저장되는 곳 (`Local` / `Remote`)

```text
[Working Directory] --- (git add) ---> [Staging Area] --- (git commit) ---> [Local Repository] --- (git push) ---> [Remote Repository]
                                                                                              <--- (git pull) ---
```

### 🔨 로컬 저장소 주요 명령어

```bash
# 1. 현재 폴더를 Git 로컬 저장소로 초기화 (.git 숨김 폴더 생성)
git init

# 2. 현재 파일들의 상태 확인 (Untracked, Modified, Staged 등)
git status

# 3. 변경된 파일을 Staging Area에 추가
git add <파일명>       # 특정 파일만 추가
git add .            # 모든 변경 사항 추가

# 4. 변경 사항을 하나의 버전(스냅샷)으로 기록
git commit -m "커밋 메시지"
git commit -am "커밋 메시지"   # 추적(Tracked) 중인 파일의 add + commit 동시 수행

# 5. 변경 내용 및 커밋 이력 확인
git diff             # 최근 커밋과 현재 작업 디렉토리의 차이점 비교
git log              # 전체 커밋 이력 상세 조회
git log --oneline    # 커밋 이력을 한 줄씩 요약 조회
```

### ⏪ Git 복구 & 실수 되돌리기 (`restore` / `reset`)

상황에 따라 작업 내용을 되돌리거나 커밋을 취소할 수 있습니다.

| 상황 | 해결 명령어 | 비고 |
| :--- | :--- | :--- |
| 수정한 파일을 마지막 커밋 상태로 되돌림 | `git restore <파일명>` | 작업 내용 폐기 |
| 모든 파일의 수정 사항을 한 번에 취소 | `git restore .` | 전체 작업 내용 폐기 |
| `git add` 취소 (Staging 해제) | `git restore --staged <파일명>` | 첫 커밋 전에는 `git rm --cached <파일명>` |
| 커밋만 취소 (Staging + 작업물 유지) | `git reset --soft <커밋ID>` | 커밋 메시지만 고치고 싶을 때 유용 |
| 커밋 + Staging 취소 (작업물 유지) | `git reset <커밋ID>` | `--mixed` 옵션과 동일 (기본값) |
| 커밋 + Staging + 작업물 모두 완전 삭제 | `git reset --hard <커밋ID>` | ⚠️ **주의:** 복구 불가 |

> ⚠️ **주의:** `git reset`은 원격 저장소(`GitHub`)에 공유(`push`)되지 않은 로컬 상태에서만 사용하는 것을 권장합니다.

### 🌿 Branch(브랜치) 기초 및 이동

```bash
# 특정 커밋 시점으로 임시 이동 (detached HEAD 상태)
git checkout <커밋ID>

# 새 브랜치 생성과 동시에 이동
git switch -c <브랜치명>

# 기존 브랜치로 전환
git switch <브랜치명>

# 로컬 브랜치 목록 확인
git branch -l
```

### ☁️ GitHub 원격 저장소 연동 및 협업

1. **Personal Access Token (PAT) 발급:**
   * GitHub `Settings` → `Developer settings` → `Personal access tokens (classic)` → `Generate new token`
   * `Note` 작성, 만료기간 설정, **`repo` 권한 전체 체크** 후 토큰 생성 및 복사
2. **원격 저장소 연결 및 업로드 (`push`):**

   ```bash
   # 원격 저장소 주소 연결 (별칭: origin)
   git remote add origin https://github.com/사용자이름/저장소이름.git
   git remote -v   # 연결 확인
   
   # 기본 브랜치를 main으로 설정 후 업로드
   git branch -M main
   git push -u origin main   # -u 옵션 설정 이후에는 'git push'만 입력 가능
   ```

3. **다른 환경에서 이어하기 & 협업 (`clone` / `pull`):**
   * 팀원 초대: GitHub 저장소 `Settings` → `Collaborators` → 팀원 추가

   ```bash
   # 원격 저장소 프로젝트를 내 컴퓨터로 통째로 복제해오기
   git clone https://github.com/사용자이름/저장소이름.git
   
   # 원격 저장소의 최신 변경 사항을 로컬로 받아와 병합
   git pull
   ```

> **💡 협업 TIP:** 팀 프로젝트에서는 `push`보다 **`pull`을 먼저 습관화**하는 것이 중요합니다. 다른 팀원이 먼저 올린 코드가 있는지 확인하고 받아와야 충돌(Conflict)을 예방할 수 있습니다.

---

## 4. Git & GitHub 심화 (Branch, Conflict, PR)

### 🌳 브랜치 전략 개념

브랜치를 분리하여 개발하면 코드 충돌을 최소화하고 안정적인 협업이 가능합니다.

* **`main`:** 최종 릴리즈된 안정적인 브랜치 (항상 존재)
* **`develop` (`dev`):** 다음 출시 버전을 위해 개발이 진행되는 브랜치 (항상 존재)
* **`feature`:** 특정 새로운 기능을 개발하기 위한 이벤트성 브랜치 (개발 완료 후 병합 및 삭제)

### ⚔️ 브랜치 병합(`merge`)과 충돌(`Conflict`) 해결

1. **병합 명령어:**

   ```bash
   git switch main      # 기준이 될 브랜치로 이동
   git merge dev        # dev 브랜치의 작업 내용을 main에 병합
   ```

2. **충돌(Conflict) 발생 및 해결:**
   * 두 브랜치에서 **같은 파일의 같은 줄**을 서로 다르게 수정했을 때 발생합니다.
   * 충돌이 발생한 파일을 열면 아래와 같이 표시됩니다:

     ```text
     <<<<<<< HEAD (Current Change)
     현재 브랜치(main)의 코드 내용
     =======
     병합하려는 브랜치(dev)의 코드 내용
     >>>>>>> dev (Incoming Change)
     ```

   * **해결 방법:** VS Code의 편의 버튼(`Accept Current/Incoming/Both Change`)을 활용하거나 직접 코드를 수정하고 특수 마커(`<<<<<<<`, `=======`, `>>>>>>>`)를 지운 뒤 다시 커밋합니다.

     ```bash
     git add .
     git commit -m "충돌 해결"
     git push
     ```

### 📬 Pull Request (PR) 및 코드 리뷰 과정

1. 기능 브랜치(`dev` 등)에서 작업 완료 후 원격 저장소로 `git push` 합니다.
2. GitHub 저장소의 **Pull requests** 탭에서 `New pull request`를 클릭합니다.
3. `base: main` ← `compare: dev` 방향으로 설정하고 변경 사항을 확인합니다.
4. PR 제목과 설명(마크다운 지원)을 작성하고 `Create pull request`를 누릅니다.
5. 팀원들과 코드 리뷰 및 코멘트(`LGTM 👍` 등)를 주고받은 뒤 `Merge pull request`를 클릭하여 병합을 완료합니다.

---

## 5. 실전 협업을 위한 팁

### 🙈 `.gitignore` - Git 제외 파일 관리

프로젝트 루트 폴더에 `.gitignore` 파일을 생성하면, Git이 추적하지 않을 파일이나 폴더를 지정할 수 있습니다.

* **Git으로 버전 관리하면 안 되는 주요 파일:**

| 예시 | 제외해야 하는 이유 |
| :--- | :--- |
| `node_modules/` | 파일 수가 많고 용량이 매우 크며, `package.json`을 통해 언제든 재설치 가능 |
| `.env` | DB 비밀번호, API 키, 토큰 등 보안상 민감한 정보 포함 |
| `*.log` | 실행 기록용 파일이므로 소스 코드 버전 관리 대상이 아님 |
| `.vscode/` | 에디터 설정 파일은 개발자 개인마다 환경이 다름 |

* **`.gitignore` 작성 예시:**

  ```gitignore
  # 의존성 패키지
  node_modules/

  # 환경 변수 (민감 정보)
  .env

  # IDE 설정
  .vscode/

  # 로그 파일
  *.log
  ```

### 📝 Markdown(마크다운) 핵심 문법

가독성과 간결성이 뛰어나 GitHub의 `README.md`, Issue, PR 작성에 필수적으로 사용됩니다.

* **제목:** `# 큰 제목`, `## 중간 제목`, `### 작은 제목`
* **목록:**
  * 순서 있는 목록: `1. 항목`
  * 순서 없는 목록: `- 항목` (하위 목록은 `Tab`, 상위로 복귀는 `Shift + Tab`)
* **강조:** `*이탤릭체*`, `**볼드체**`, `~~취소선~~`
* **코드:**
  * 인라인 코드: `` `코드` ``
  * 코드 블록: 백틱(`` ` ``) 3개로 감싸기
* **링크 및 이미지:**
  * 링크: `[표시할 텍스트](링크 URL)`
  * 이미지: `![대체 텍스트](이미지 URL)`
* **표(Table):** 파이프(`|`)와 하이픈(`-`)을 사용하여 열과 행 구분

### 🎨 GitHub 프로필 & 프로젝트 꾸미기

* **프로필 README 만들기:** 본인의 GitHub `Username`과 **동일한 이름의 Public Repository**를 생성하고 `README.md`를 작성하면 프로필 메인 상단에 소개글이 노출됩니다.
* **프로젝트 README 추천 구성:** `New Features`(새 기능), `Bug Fixes`(버그 수정), `Documentation`(문서화), `Dependency Upgrades`(의존성 업데이트), `Contributors`(기여자) 등.
* **잔디 심기 (Contributions):** 꾸준한 커밋과 PR 활동은 성실한 개발자의 척도가 됩니다.

### 📌 Commit Convention (커밋 컨벤션)

협업 시 커밋 메시지만 보고도 작업의 목적을 한눈에 파악할 수 있도록 통일된 접두사를 사용합니다.

| 태그 | 설명 |
| :--- | :--- |
| **`Feat`** | 새로운 기능을 추가한 경우 |
| **`Fix`** | 버그를 수정한 경우 |
| **`HOTFIX`** | 급하게 치명적인 버그를 고쳐야 하는 경우 |
| **`Style`** | CSS 등 사용자 UI 디자인을 변경한 경우 |
| **`Refactor`** | 프로덕션 코드를 리팩토링한 경우 |
| **`Docs`** | 문서를 수정한 경우 (`README.md` 등) |
| **`Comment`** | 필요한 주석을 추가 및 변경한 경우 |
| **`Rename`** | 파일 혹은 폴더명을 수정하거나 위치를 옮긴 경우 |
| **`Remove`** | 파일을 삭제하는 작업만 수행한 경우 |

### 💬 Code Review (코드 리뷰) & 개발자 줄임말

* **목적:** 기술 부채 감소, 코드의 부작용 및 오류 사전 방지, 팀원 간 지식 공유 (*"리뷰할 게 없다면 칭찬하기!"*)
* **자주 쓰이는 영어 약어:**
  * **`LGTM`**: **L**ooks **G**ood **T**o **M**e (좋아 보입니다 / 승인)
  * **`FYI`**: **F**or **Y**our **I**nformation (참고로 알려드립니다)
  * **`IMO` / `IMHO`**: **I**n **M**y (**H**umble) **O**pinion (제 (부족한) 의견으로는)
  * **`AFAIK`**: **A**s **F**ar **A**s **I** **K**now (내가 아는 한)
  * **`GOTCHA`**: I've **Got** **You** (이해했습니다 / 알겠습니다)
  * **`SSIA`**: **S**ubject **S**ays **I**t **A**ll (제목이 곧 내용)
  * **`TIA`**: **T**hanks **I**n **A**dvance (미리 감사합니다)