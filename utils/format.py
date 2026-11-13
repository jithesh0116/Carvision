def format_money(value, symbol="₹", locale="en-IN"):
    """8,45,000 style grouping for en-IN, 845,000 otherwise. Presentation only."""
    n = int(round(float(value)))
    s = str(abs(n))
    if locale == "en-IN" and len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    else:
        s = f"{abs(n):,}"
    return f"{'-' if n < 0 else ''}{symbol}{s}"


def round_price(value, step):
    step = max(int(step), 1)
    return float(round(float(value) / step) * step) if value >= step * 10 else float(round(value))
