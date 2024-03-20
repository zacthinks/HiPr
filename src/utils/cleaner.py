import re


def cleaner(text,
            verbose=False,
            simpleurl_p=re.compile(r'\S*\.[a-zA-Z]{2,6}/\S*'),
            dash_p=re.compile('([a-zA-Z]{2,})- (a|the) '),
            hyphen_p=re.compile('([a-zA-Z]{2,})- ([a-zA-Z]{2,})'),
            footnote_p=re.compile(r'([a-zA-Z]{2,}[.,;?"])(\[?\d+\]?)'),
            slashspace_p=re.compile(r'([a-zA-Z]+/) ([a-zA-Z]+)'),
            numseries_p=re.compile(r'(\d+ \d+)(\s+\d+)+ (\d+ \d+)')
            ):
    def repl_func(match, repl, pattern, supress_verbose=False):
        result = match.expand(repl)
        if verbose and not supress_verbose:
            print(f"{pattern}: '{match.group()}' -> '{result}'")
        return result
    text = simpleurl_p.sub(lambda m: repl_func(m, '<URL>', 'simpleurl'), text)
    text = hyphen_p.sub(lambda m: repl_func(m, r'\1\2', 'hyphen_remover'), text)
    text = dash_p.sub(lambda m: repl_func(m, r'\1 - a ', 'dash_spacer'), text)
    text = footnote_p.sub(lambda m: repl_func(m, r'\1', 'footnote_num_remover'), text)
    text = slashspace_p.sub(lambda m: repl_func(m, r'\1\2', 'slashspace_remover'), text)
    text = numseries_p.sub(lambda m: repl_func(m, r'\1 ...<NUMBERS>... \2', 'num_series_truncator'), text)

    return text
