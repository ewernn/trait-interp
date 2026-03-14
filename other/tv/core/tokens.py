"""Token-level span extraction utilities.

split_into_clauses: split tokens at punctuation boundaries with mean scores
extract_window_spans: sliding window max-activating span finder
"""


def split_into_clauses(tokens, scores):
    """Split tokens at punctuation boundaries with per-clause mean score.

    Args:
        tokens: list of token strings
        scores: list of float scores (one per token)

    Returns:
        list of {text, token_range, mean_score}
    """
    separators = {',', '.', ';', '\u2014', '\n', '!', '?', ':'}
    clauses = []
    current_tokens = []
    current_scores = []
    start_idx = 0

    n = min(len(tokens), len(scores))
    for i in range(n):
        tok, score = tokens[i], scores[i]
        current_tokens.append(tok)
        current_scores.append(score)

        is_sep = any(s in tok for s in separators)
        is_last = (i == n - 1)

        if (is_sep or is_last) and current_scores:
            text = ''.join(current_tokens)
            if text.strip():
                clauses.append({
                    'text': text,
                    'token_range': [start_idx, i + 1],
                    'mean_score': sum(current_scores) / len(current_scores),
                })
            current_tokens = []
            current_scores = []
            start_idx = i + 1

    return clauses


def extract_window_spans(tokens, scores, window_length, top_k=10):
    """Sliding window with non-overlapping greedy selection.

    Args:
        tokens: list of token strings
        scores: list of float scores (one per token)
        window_length: number of tokens per window
        top_k: max number of spans to return

    Returns:
        list of {text, token_range, mean_score} sorted by |mean_score| descending
    """
    n = min(len(tokens), len(scores))
    if n == 0:
        return []

    wl = min(window_length, n)

    # Running sum for O(n) sliding window
    running_sum = sum(scores[:wl])
    windows = [(abs(running_sum), running_sum / wl, 0)]
    for i in range(1, n - wl + 1):
        running_sum += scores[i + wl - 1] - scores[i - 1]
        windows.append((abs(running_sum), running_sum / wl, i))

    # Sort by |mean_score| descending
    windows.sort(key=lambda x: x[0], reverse=True)

    # Greedy non-overlapping selection
    used = set()
    spans = []
    for abs_score, mean_score, start in windows:
        if len(spans) >= top_k:
            break
        positions = set(range(start, start + wl))
        if positions & used:
            continue
        used |= positions
        text = ''.join(tokens[start:start + wl])
        if not text.strip():
            continue
        spans.append({
            'text': text,
            'token_range': [start, start + wl],
            'mean_score': mean_score,
        })

    return spans
