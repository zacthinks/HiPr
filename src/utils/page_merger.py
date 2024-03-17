import re


def get_max_match(text, loc, delim, window=100, threshold=3):
    left = text[max(0, loc - window):loc]
    right = text[loc + len(delim):loc + len(delim) + window]
    left_str = r"\s*"
    right_str = r"\s*"
    match_count = 0

    while True:
        next_block = re.search(r"(\S+)" + left_str + "$", left).group(1)
        if next_block.isdigit():
            new_left_str = r"\s*\d+" + left_str
        else:
            new_left_str = r"\s*" + re.escape(next_block) + left_str

        new_match_count = len(re.findall(new_left_str + delim, text, re.IGNORECASE))
        if new_match_count >= threshold:
            left_str = new_left_str
            match_count = new_match_count
        else:
            break

    while True:
        next_block = re.search("^" + right_str + r"(\S+)", right).group(1)
        if next_block.isdigit():
            new_right_str = right_str + r"\d+\s*"
        else:
            new_right_str = right_str + re.escape(next_block) + r"\s*"

        new_match_count = len(re.findall(left_str + delim + new_right_str, text, re.IGNORECASE))
        if new_match_count >= threshold:
            right_str = new_right_str
            match_count = new_match_count
        else:
            break

    if match_count > 0:
        return re.compile(left_str + delim + right_str, re.IGNORECASE), match_count
    else:
        return None


def crude_headerfooter_matcher(
        text, loc, delim, window=100, min_len=3,
        left_p=re.compile(r"\b(?=[A-Z0-9])[^a-z]*$"),
        right_p=re.compile(r"^[^a-z]*\b")):
    d_len = len(delim)
    left = text[loc - window:loc]
    right = text[loc + d_len:loc + d_len + window]
    if left_match := left_p.search(left):
        left_len = len(left_match.group())
    else:
        left_len = 0
    right_len = len(right_p.search(right).group())

    if left_len < min_len:
        left_len = 0
    if right_len < min_len:
        right_len = 0
    if left_len + right_len > 0:
        return loc - left_len, loc + d_len + right_len
    else:
        return None


def join_pages(pages,
               delim="@@@PAGE@@@",
               delim_regex=re.compile('@@@PAGE@@@', re.IGNORECASE),
               use_crude=True,
               verbose=False):
    def repl_func(matchobj):
        nonlocal first
        if verbose:
            print("\tDeleted by get_max_match: ", matchobj.group())
        if first:
            return " "
            first = False
        else:
            return delim

    joined = delim.join(pages)
    n_delim = len(pages) - 1
    while len(delim_regex.findall(joined)) > n_delim:
        delim = f"@{delim}@"
        delim_regex = re.compile(delim)
        joined = delim.join(pages)

    while match := delim_regex.search(joined):
        loc = match.start()
        first = True
        if max_match := get_max_match(joined, loc, delim):
            if verbose:
                print(f"{max_match[1]} matches ({max_match[1] / n_delim:.2%}) found by get_max_match to be deleted:")
            joined = max_match[0].sub(repl_func, joined)
        elif use_crude and (crude_match := crude_headerfooter_matcher(joined, loc, delim)):
            if verbose:
                print("Crude match to be deleted: ", joined[crude_match[0]:crude_match[1]])
            joined = joined[:crude_match[0]] + " " + joined[crude_match[1]:]
        else:
            joined = joined[:loc] + " " + joined[loc + len(delim):]

    return joined
