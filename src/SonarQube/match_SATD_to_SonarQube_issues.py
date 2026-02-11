import json
import pandas as pd

SATD_FILE = "fixed_code_comments_with_priority.jsonl"
SONAR_FILE = "sonarqube_all_projects.json"
OUTPUT_CSV = "satd_sonar_matched.csv"
LINE_TOLERANCE = 2  # lines above/below to consider a match

# Load SATD data
satd_entries = []
with open(SATD_FILE, 'r') as f:
    for line in f:
        s = json.loads(line)
        if s['satd_label'] != 'non_debt':
            satd_entries.append(s)

print(f"Filtered SATD entries (non_non_debt): {len(satd_entries)}")

# Load SonarQube data
with open(SONAR_FILE, 'r') as f:
    sonar_data = json.load(f)

sonar_issues = sonar_data.get('issues', [])
print(f"Loaded {len(sonar_issues)} SonarQube issues")

# Matching
matched_pairs = []

for s in satd_entries:
    # Normalize project name
    s_project = s['repo_name'].split('/')[-1]

    for si in sonar_issues:
        si_project = si.get('project_key', si.get('project'))
        if s_project != si_project:
            continue

        # Normalize paths for suffix matching
        s_first_file = s['first_file'].lstrip("./")
        s_last_file = s['last_file'].lstrip("./")
        si_file = si['component'].split(":",1)[-1].lstrip("./")  # remove project prefix

        file_match = si_file.endswith(s_first_file) or si_file.endswith(s_last_file)
        if not file_match:
            continue

        # Line match with tolerance
        if 'line' in si:
            si_line = si['line']
            if not (s['first_line'] - LINE_TOLERANCE <= si_line <= s['last_line'] + LINE_TOLERANCE):
                continue

        # Match found
        matched_pairs.append({
            'satd_comment': s['comment'],
            'satd_embedding_score': s['priority_features']['embedding'],
            'satd_file': s['first_file'],
            'satd_line_start': s['first_line'],
            'satd_line_end': s['last_line'],
            'sonar_project': si_project,
            'sonar_component': si['component'],
            'sonar_line': si.get('line'),
            'sonar_severity': si.get('severity'),
            'sonar_debt': si.get('debt'),
            'sonar_type': si.get('type'),
            'sonar_message': si.get('message')
        })

print(f"Found {len(matched_pairs)} matched SATD-SonarQube pairs")

# Save CSV
df = pd.DataFrame(matched_pairs)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved matched SATD-Sonar data to {OUTPUT_CSV}")
