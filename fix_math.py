import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    def replacer(match):
        math_content = match.group(0)
        # We only want to replace _ with \sb inside the math block
        return math_content.replace('_', r'\sb ')

    # Match $$...$$
    new_content = re.sub(r'(?<!\\)\$\$(.*?)(?<!\\)\$\$', replacer, content, flags=re.DOTALL)
    # Match $...$
    new_content = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', replacer, new_content, flags=re.DOTALL)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed {filepath}')
    else:
        print(f'No changes in {filepath}')

process_file(r'e:\NAM_BA\NGHIEN_CUU_KHOA_HOC\DiffMM_Overview.md')
process_file(r'e:\NAM_BA\NGHIEN_CUU_KHOA_HOC\Flow_Matching_1_Overview.md')
