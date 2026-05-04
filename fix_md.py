import os
import textwrap
import re

def parse_markdown_table(lines):
    # Returns (headers, rows)
    headers = [c.strip() for c in lines[0].strip('|').split('|')]
    rows = []
    for line in lines[2:]:
        if not line.strip(): continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        # padding in case cells are missing
        cells += [''] * (len(headers) - len(cells))
        rows.append(cells[:len(headers)])
    return headers, rows

def wrap_cell(cell_text):
    lines = textwrap.wrap(cell_text, width=60, break_long_words=False)
    if not lines: return ""
    return "<br>\n      ".join(lines)

def table_to_html(headers, rows):
    html = ["<table>", "  <thead>", "    <tr>"]
    for h in headers:
        html.append(f"      <th>{wrap_cell(h)}</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")
    for row in rows:
        html.append("    <tr>")
        for cell in row:
            html.append(f"      <td>{wrap_cell(cell)}</td>")
        html.append("    </tr>")
    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)

def process_file(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove backticks
    content = content.replace('`', '')
    
    # 2. Fix wide tables
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Very basic markdown table detection
        if stripped.startswith('|') and stripped.endswith('|'):
            # gather table lines
            table_lines = []
            j = i
            while j < len(lines) and lines[j].strip().startswith('|') and lines[j].strip().endswith('|'):
                table_lines.append(lines[j])
                j += 1
            
            # Check if it's a valid table
            if len(table_lines) >= 2 and '---' in table_lines[1]:
                # Is it wide?
                if any(len(tl) > 160 for tl in table_lines):
                    headers, rows = parse_markdown_table(table_lines)
                    html_table = table_to_html(headers, rows)
                    new_lines.extend(html_table.split('\n'))
                else:
                    new_lines.extend(table_lines)
            else:
                new_lines.extend(table_lines)
            i = j
        else:
            new_lines.append(line)
            i += 1
            
    new_content = '\n'.join(new_lines)
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)

files = []
for root, dirs, filenames in os.walk('.'):
    if '.venv' in root or '.git' in root or '.pytest_cache' in root:
        continue
    for filename in filenames:
        if filename.endswith('.md'):
            files.append(os.path.join(root, filename))

for f in files:
    process_file(f)
print("Done.")
