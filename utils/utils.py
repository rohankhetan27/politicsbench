import json
def robust_json_loads(json_str: str) -> dict:
    """
    Mimics json.loads but repairs malformed JSON with unescaped newlines in string values
    if parsing fails initially.
    """
    def repair_json_newlines(s: str) -> str:
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(s):
            char = s[i]

            if in_string:
                if escape_next:
                    result.append(char)
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                    result.append(char)
                elif char == '"':
                    in_string = False
                    result.append(char)
                elif char in '\n\r':
                    result.append('\\n')
                else:
                    result.append(char)
            else:
                if char == '"':
                    in_string = True
                result.append(char)

            i += 1

        return ''.join(result)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = repair_json_newlines(json_str)
        return json.loads(repaired)