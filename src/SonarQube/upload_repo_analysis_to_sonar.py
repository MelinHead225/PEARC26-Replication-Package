import subprocess
from pathlib import Path

# CONFIG
SONAR_SCANNER_PATH = r"C:\Users\sonar-scanner.bat"
SONAR_HOST = "http://localhost:9000"
SONAR_TOKEN = ""

REPOS_DIR = Path("repos")
EXCLUSIONS = "**/build/**,**/vendor/**,**/.git/**"

# RUN ANALYSIS PER REPO
for repo in REPOS_DIR.iterdir():
    if not repo.is_dir():
        continue

    print(f"\n=== Analyzing {repo.name} ===")

    cmd = [
        SONAR_SCANNER_PATH,
        f"-Dsonar.projectKey={repo.name}",
        f"-Dsonar.projectName={repo.name}",
        "-Dsonar.sources=.",
        f"-Dsonar.host.url={SONAR_HOST}",
        f"-Dsonar.login={SONAR_TOKEN}",
        "-Dsonar.scm.disabled=true",
        "-Dsonar.cpd.disabled=true",        
        f"-Dsonar.exclusions={EXCLUSIONS}"
    ]

    try:
        subprocess.run(cmd, cwd=repo, check=True)
        print(f"Analysis completed for {repo.name}")
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed for {repo.name}!")
        print("Return code:", e.returncode)
        print("Command:", " ".join(e.cmd))
