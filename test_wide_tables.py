import os
import re

def find_wide_tables():
    files = []
    for root, dirs, filenames in os.walk('.'):
        if '.venv' in root or '.git' in root or '.pytest_cache' in root:
            continue
        for filename in filenames:
            if filename.endswith('.md'):
                files.append(os.path.join(root, filename))
    
    wide_tables = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_table = False
        table_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                in_table = True
                table_lines.append(line.rstrip('\n'))
            else:
                if in_table:
                    if any(len(tl) > 160 for tl in table_lines):
                        wide_tables.append((fpath, table_lines))
                    in_table = False
                    table_lines = []
        if in_table:
            if any(len(tl) > 160 for tl in table_lines):
                wide_tables.append((fpath, table_lines))
    return wide_tables

wt = find_wide_tables()
for f, lines in wt:
    print(f"File: {f} has a wide table with {len(lines)} lines (max len {max(len(l) for l in lines)})")

