import subprocess
import os
import sys

def get_sort_key(path):
    p = path.replace('\\', '/')
    if p in ['.gitignore', 'requirements.txt', 'AGENTS.md', 'README.md', 'FarmWatch_Model_Audit_Report.md']:
        return (0, p)
    elif p.startswith('docs/') or p.startswith('presentation/'):
        return (1, p)
    elif p.startswith('data/'):
        return (2, p)
    elif p.startswith('models/'):
        return (3, p)
    elif p.startswith('src/utils/'):
        return (4, p)
    elif p.startswith('src/models/'):
        return (5, p)
    elif p.startswith('src/training/'):
        return (6, p)
    elif p.startswith('src/explainability/'):
        return (7, p)
    elif p.startswith('src/inference/'):
        return (8, p)
    elif p.startswith('src/'):
        return (9, p)
    elif p.startswith('backend/'):
        return (10, p)
    elif p.startswith('frontend/'):
        return (11, p)
    elif p.startswith('notebooks/'):
        return (12, p)
    elif p.startswith('reports/'):
        return (13, p)
    elif p.startswith('artifacts/'):
        return (14, p)
    elif p.startswith('scripts/'):
        return (15, p)
    elif p.startswith('tests/'):
        return (16, p)
    return (17, p)

def get_commit_msg(path):
    p = path.replace('\\', '/')
    name = os.path.basename(p)
    if p == '.gitignore':
        return 'chore: add .gitignore configuration'
    elif p == 'requirements.txt':
        return 'chore: add requirements.txt dependencies'
    elif p == 'AGENTS.md':
        return 'docs: add AGENTS.md project guidelines and verification standards'
    elif p == 'README.md':
        return 'docs: add comprehensive README.md'
    elif p == 'FarmWatch_Model_Audit_Report.md':
        return 'docs: add FarmWatch_Model_Audit_Report.md'
    elif p.startswith('docs/plans/'):
        return f'docs(plans): add {name}'
    elif p.startswith('docs/'):
        return f'docs: add {name}'
    elif p.startswith('presentation/'):
        return f'docs(presentation): add {name}'
    elif p.startswith('data/manifests/'):
        return f'data(manifests): add {name} split manifest'
    elif p.startswith('data/splits/'):
        return f'data(splits): add {name} metadata'
    elif p.startswith('models/'):
        return f'feat(models): add {name} configuration'
    elif p.startswith('src/models/'):
        return f'feat(models): add {name}'
    elif p.startswith('src/training/'):
        return f'feat(training): add {name}'
    elif p.startswith('src/explainability/'):
        return f'feat(explainability): add {name}'
    elif p.startswith('src/inference/'):
        return f'feat(inference): add {name}'
    elif p.startswith('src/utils/'):
        return f'feat(utils): add {name}'
    elif p.startswith('src/'):
        return f'feat(src): add {name}'
    elif p.startswith('backend/app/api/'):
        return f'feat(api): add {name}'
    elif p.startswith('backend/app/services/'):
        return f'feat(services): add {name}'
    elif p.startswith('backend/app/schemas/'):
        return f'feat(schemas): add {name}'
    elif p == 'backend/app/main.py':
        return 'feat(backend): add FastAPI application main entrypoint'
    elif p.startswith('backend/'):
        return f'feat(backend): add {name}'
    elif p.startswith('frontend/src/components/'):
        return f'feat(ui): add {name} component'
    elif p.startswith('frontend/src/types/'):
        return f'feat(frontend): add {name} TypeScript definitions'
    elif p.startswith('frontend/src/lib/'):
        return f'feat(frontend): add {name} client library'
    elif p.startswith('frontend/src/assets/'):
        return f'assets(frontend): add {name}'
    elif p.startswith('frontend/public/'):
        return f'assets(frontend): add {name}'
    elif p.startswith('frontend/src/'):
        return f'feat(frontend): add {name}'
    elif p.startswith('frontend/'):
        return f'chore(frontend): add {name}'
    elif p.startswith('notebooks/'):
        return f'feat(notebooks): add {name}'
    elif p.startswith('reports/model_audit/error_analysis/'):
        return f'report(audit): add error analysis artifact {name}'
    elif p.startswith('reports/model_audit/'):
        return f'report(audit): add {name}'
    elif p.startswith('reports/'):
        return f'report: add {name}'
    elif p.startswith('artifacts/'):
        return f'artifacts: add {name}'
    elif p.startswith('scripts/'):
        return f'chore(scripts): add {name}'
    elif p.startswith('tests/'):
        return f'test: add {name}'
    return f'feat: add {name}'

def main():
    res = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    files.sort(key=get_sort_key)
    
    total = len(files)
    print(f'Starting individual commit of {total} files...')
    
    for idx, f in enumerate(files, 1):
        msg = get_commit_msg(f)
        # Stage only this one file
        add_res = subprocess.run(['git', 'add', f], capture_output=True, text=True)
        if add_res.returncode != 0:
            print(f'Failed to git add {f}: {add_res.stderr}')
            sys.exit(1)
        
        # Commit this single file
        commit_res = subprocess.run(['git', 'commit', '-m', msg], capture_output=True, text=True)
        if commit_res.returncode != 0:
            print(f'Failed to commit {f}: {commit_res.stderr}')
            sys.exit(1)
        
        print(f'[{idx}/{total}] {f} -> {msg}')

    print('All files successfully committed!')

if __name__ == '__main__':
    main()
