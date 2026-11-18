"""Exact CPU tokenizer accounting; does not load model weights or run generation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

from .schema import write_json
from .tokenization import TokenCounter

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    root = Path(root)
    base = root / 'runs/baseline_bootstrap_v2'
    output = root / 'runs/simplemem_mechanism_v1/storage_tokens.json'
    public = root / 'research/baseline_bootstrap_v2/public/conversations.json'
    source = root / 'third_party/simplemem_official/simplemem/core/answer_generator.py'
    text = source.read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'AnswerGenerator')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_format_contexts')
    namespace = {}
    # Execute the exact pinned formatter without importing its model/database dependencies.
    exec(compile('from __future__ import annotations\n' +
                 textwrap.dedent(ast.get_source_segment(text, node)), str(source), 'exec'), namespace)
    counter = TokenCounter(root / 'models/Qwen3.5-9B')
    rows = []
    for conv in json.loads(public.read_text(encoding='utf-8')):
        cid = conv['conversation_id']
        path = base / cid / 'memories.json'
        if not path.exists():
            rows.append({'conversation_id': cid, 'status': 'pending_memory_completion'})
            continue
        entries = json.loads(path.read_text(encoding='utf-8'))['entries']
        raw = '\n'.join(f"[{t['timestamp']}] {t['speaker']}: {t['content']}" for t in conv['turns'])
        content = '\n'.join(e['lossless_restatement'] for e in entries)
        formatted = namespace['_format_contexts'](None, [SimpleNamespace(**e) for e in entries])
        raw_tokens, content_tokens, formatted_tokens = (counter.count(s) for s in (raw, content, formatted))
        rows.append({'conversation_id': cid, 'status': 'complete', 'turns': len(conv['turns']),
            'entries': len(entries), 'raw_timestamped_transcript_tokens': raw_tokens,
            'memory_content_only_tokens': content_tokens,
            'memory_official_formatted_tokens': formatted_tokens,
            'memory_json_bytes': path.stat().st_size,
            'content_to_raw_token_ratio': content_tokens/raw_tokens if raw_tokens else None,
            'formatted_to_raw_token_ratio': formatted_tokens/raw_tokens if raw_tokens else None,
            'memory_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'tokenizer': 'Qwen3.5-9B; local exact tokenizer; add_special_tokens=False',
        'formatter_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'model_weights_loaded': False, 'generation_calls': 0, 'conversations': rows,
        'interpretation': 'Full corpus text/format accounting, not query-stage tokens or answer quality.',
        'exclusions': 'Vector index/embedding storage and source cold storage are not included in memory_json_bytes.'}
    write_json(output, result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False))
