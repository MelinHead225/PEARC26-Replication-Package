REPOS_TO_PROCESS = [    
    'ornladios/ADIOS2',
    'visit-dav/visit',
    'dyninst/dyninst',
    'UO-OACISS/tau2',
    'hypre-space/hypre',
    'trilinos/Trilinos',
    'kokkos/kokkos',
    'StanfordLegion/legion',
    'spack/spack'
]

def _build_clone_url(repo: str, use_ssh: bool) -> str:
    owner_repo = repo.strip()
    if use_ssh:
        return f"git@github.com:{owner_repo}.git"
    return f"https://github.com/{owner_repo}.git"


def clone_repos(repos, target_dir='repos', dry_run=False, force=False, use_ssh=False):
    """Clone all repositories in `repos` into `target_dir`.

    Args:
        repos (iterable): iterable of 'owner/repo' strings
        target_dir (str or Path): directory to place cloned repos
        dry_run (bool): if True, print commands but don't execute
        force (bool): if True, remove existing target directory before cloning
        use_ssh (bool): if True, use SSH clone URL instead of HTTPS
    """
    from pathlib import Path
    import shutil
    import subprocess

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for repo in repos:
        repo = repo.strip()
        if not repo or repo.startswith("#"):
            continue
        repo_name = repo.split("/")[-1]
        dest = target / repo_name

        if dest.exists():
            if force:
                if dry_run:
                    print(f"[dry-run] would remove existing '{dest}'")
                else:
                    print(f"Removing existing directory '{dest}'")
                    shutil.rmtree(dest)
            else:
                print(f"Skipping '{repo}': target '{dest}' already exists. Use --force to reclone.")
                continue

        url = _build_clone_url(repo, use_ssh)
        cmd = ["git", "clone", url, str(dest)]

        if dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            continue

        print(f"Cloning {repo} -> {dest}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo}: {e}")


def main():
    target = "repos"
    dry_run = False
    force = False
    use_ssh = False

    clone_repos(REPOS_TO_PROCESS, target_dir=target, dry_run=dry_run, force=force, use_ssh=use_ssh)


if __name__ == '__main__':
    main()