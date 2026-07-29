import shutil
from pathlib import Path

from src.config import MODEL_SAVE_PATH

ROOT_DIR = Path(".")
DEPLOY_DIR = Path("deploy")
DIST_DIR = Path("dist/huggingface_space")
SPACE_REPO = "fcitarelli/twitter-sentiment-analysis"


def build_package():
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    shutil.copy(DEPLOY_DIR / "app.py", DIST_DIR / "app.py")
    shutil.copy(
        ROOT_DIR / "requirements.txt",
        DIST_DIR / "requirements.txt",
    )
    shutil.copy(DEPLOY_DIR / "README_space.md", DIST_DIR / "README.md")

    model_src = Path(MODEL_SAVE_PATH)
    if model_src.is_dir():
        shutil.copytree(model_src, DIST_DIR / "model")
    else:
        print(f"[deploy] WARNING: {model_src} not found, skipping model copy")

    return DIST_DIR


def simulate_push(package_dir):
    print("[deploy] Simulated Hugging Face Space deploy (no token used)")
    print(f"[deploy] Package ready at: {package_dir.resolve()}")
    print(f"[deploy] Would run: huggingface-cli upload {SPACE_REPO} "
          f"{package_dir} --repo-type space")
    print("[deploy] Simulated push complete.")


def main():
    package_dir = build_package()
    simulate_push(package_dir)


if __name__ == "__main__":
    main()
