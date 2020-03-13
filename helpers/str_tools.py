import regex


def insert_at_index(string, index, to_insert):
    return string[:index] + to_insert + string[index:]

def remove_ugly_chars(string):
    string = string.replace("'", "").replace(" ","_").replace(".","_")
    return regex.sub(r"[\n\t\:\$ \\\!\@\#\$\%\^\&\*\(\)\-\+\=\'\"\;\<\>\?\/\~\`\.\,\;]", "_", string, regex.MULTILINE)

