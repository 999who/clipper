"""Мелочи для текстов, которые видит пользователь."""


def plural(count: int, one: str, few: str, many: str) -> str:
    """«1 слово», «3 слова», «5 слов» — с числом впереди."""
    tail = count % 100
    if 11 <= tail <= 14:
        form = many
    elif count % 10 == 1:
        form = one
    elif 2 <= count % 10 <= 4:
        form = few
    else:
        form = many
    return f"{count} {form}"
