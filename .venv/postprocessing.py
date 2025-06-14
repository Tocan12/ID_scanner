import re

def correct_text(text):

    corrections = {
        r'\bmame\b': 'nume',
        r'\bnerei\b': 'prenume',
        r'\bgrven\b': 'given',
        r'\bwatonante\b': 'nationality',
        r'\bnaitetu\b': 'nasterii',
        r'\bnp\b': 'cnp',
        r'\bdora\b': 'data',
        r'\bmaret\b': 'nasterii',
        r'\bpatbariata\b': 'expirarii',
        r'\bbat\b': 'birth',
        r'\bwat\b': 'nat',
    }

    for wrong, right in corrections.items():
        text = re.sub(wrong, right, text, flags=re.IGNORECASE)

    return text

def clean_field(value):

    return re.sub(r'[^A-Za-zĂÂÎȘȚăâîșț0-9 .-]', '', value)

def parse_fields(text):

    fields = {}
    corrected_text = correct_text(text)
    lines = corrected_text.splitlines()

    for i, line in enumerate(lines):
        line_clean = line.strip().lower()

        # skip bad lines
        if not line_clean or len(line_clean) < 3:
            continue
        if re.fullmatch(r'[^a-zăâîșț0-9 ]+', line_clean):  # only symbols
            continue


        if ':' in line:
            key, val = map(str.strip, line.split(':', 1))
            key = key.lower()
            val = clean_field(val)

            if 'nume' in key and 'prenume' not in key:
                fields['nume'] = val
            elif 'prenume' in key or 'given' in key:
                fields['prenume'] = val
            elif 'cnp' in key or 'pin' in key:
                fields['cnp'] = re.sub(r'\D', '', val)
            elif 'data nasterii' in key or 'birth' in key:
                match = re.search(r'(\d{2})[^\d]?(\d{2})[^\d]?(\d{4})', val)
                if match:
                    fields['data_nasterii'] = '.'.join(match.groups())
            continue  # skip next line lookup for this field

        if 'nume' in line_clean and 'prenume' not in line_clean:
            if i + 1 < len(lines):
                fields['nume'] = clean_field(lines[i + 1].strip())

        if 'prenume' in line_clean or 'given names' in line_clean:
            if i + 1 < len(lines):
                fields['prenume'] = clean_field(lines[i + 1].strip())

        if 'cnp' in line_clean or 'pin' in line_clean:
            if i + 1 < len(lines):
                fields['cnp'] = re.sub(r'\D', '', lines[i + 1])

        if 'data nasterii' in line_clean or 'date of birth' in line_clean:
            if i + 1 < len(lines):
                match = re.search(r'(\d{2})[^\d]?(\d{2})[^\d]?(\d{4})', lines[i + 1])
                if match:
                    fields['data_nasterii'] = '.'.join(match.groups())

    # scan whole text for a valid CNP if not found
    if 'cnp' not in fields:
        cnp_match = re.search(r'\b\d{13}\b', corrected_text)
        if cnp_match:
            fields['cnp'] = cnp_match.group()

    return fields
