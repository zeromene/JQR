"""Paper-derived uncapped IterCOMP loop; controlled development implementation."""
import json
import time
from .itercomp_primitives_v1 import sparse_weights, dual_scores, percentile_candidates, accumulate, parse_judgment


JUDGE_PROMPT = '''Given the following Question and Information, your task is to determine whether the Information alone is sufficient to answer the Question.
If the Information provides a clear and direct answer, or if the answer to the Question can be logically derived by combining multiple statements within the Information, then output: "answerable".
If the Information lacks the necessary content to fully answer the Question, then output: "unanswerable", and generate a follow-up question that identifies the most specific and essential piece of missing information required to answer the Question.
Your response must strictly follow this format:
{"answer": "answerable or unanswerable", "follow_up_question": "a specific and detailed question that would help retrieve the missing information"}
Do not include any explanation or reasoning outside of this format.'''


def units(source):
    result = []
    for p in source['paragraphs']:
        for i, s in enumerate(p['spans']):
            text = p['text'][s['start']:s['end']]
            assert text == s['text'] and text.strip()
            result.append(dict(source=p['source_id'], title=p['title'], start=s['start'], end=s['end'],
                               text=text, uid=p['source_id']+':span:'+str(i)))
    assert result and len({x['uid'] for x in result}) == len(result)
    return result


def render(candidates, selected):
    assert selected == sorted(set(selected))
    groups = {}
    for i in selected:
        c = candidates[i]
        groups.setdefault((c['source'], c['title']), []).append(c['text'])
    return '\n\n'.join('['+pid+'] '+title+'\n'+' '.join(texts) for (pid,title),texts in groups.items())


def judge_messages(question, evidence):
    return [dict(role='system', content=JUDGE_PROMPT),
            dict(role='user', content=json.dumps(dict(Question=question, Information=evidence), ensure_ascii=False))]


def compress(source, encoder, judge, record_step, max_steps=5):
    """judge returns the native result dict; record raw output before parsing or stopping."""
    assert type(max_steps) is int and 1 <= max_steps <= 5
    cs = units(source); start = time.perf_counter()
    docs_dense, docs_sparse = encoder.encode([c['text'] for c in cs])
    document_seconds = time.perf_counter() - start
    selected = []; query = source['question']; history = []
    for step in range(max_steps):
        start = time.perf_counter(); qd, qs = encoder.encode([query])
        scores = dual_scores(qd[0], qs[0], docs_dense, docs_sparse)
        found, threshold = percentile_candidates(scores)
        previous = selected; selected = accumulate(previous, found, len(cs))
        text = render(cs, selected); elapsed = time.perf_counter() - start
        messages = judge_messages(source['question'], text)
        result = judge(messages)
        row = dict(step=step, query=query, scores=scores, threshold=threshold, matched=found,
                   previous=previous, selected=selected, evidence_text=text, messages=messages,
                   result=result, filtering_seconds=elapsed)
        record_step(row)  # An invalid judge response must also remain on disk.
        if result['finish_reason'] != 'stop':
            return dict(status='judge_output_limit', steps=history+[row], selected=selected,
                        evidence_text=text, document_encoding_seconds=document_seconds)
        try: judgment = parse_judgment(result['text'])
        except (ValueError, AssertionError) as exc:
            return dict(status='judge_parse_failure', parse_error=str(exc), steps=history+[row],
                        selected=selected, evidence_text=text, document_encoding_seconds=document_seconds)
        row = dict(row, judgment=judgment); history.append(row)
        if judgment['answer'] == 'answerable':
            return dict(status='answerable', steps=history, selected=selected,
                        evidence_text=text, document_encoding_seconds=document_seconds)
        query = judgment['follow_up_question']
        # A repeated query or no new evidence is not an undocumented early-stop rule.
    return dict(status='iteration_limit', steps=history, selected=selected,
                evidence_text=text, document_encoding_seconds=document_seconds)


class BGEEncoder:
    """Frozen native BGE-M3 dense/sparse encoder, without truncation or query instruction."""
    def __init__(self, path, device='cpu', batch_size=8):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch; self.device = device; self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model, loading = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
        assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys'), loading
        self.model.to(device).eval(); self.model.requires_grad_(False)
        self.sparse = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.sparse.load_state_dict(torch.load(path/'sparse_linear.pt', map_location='cpu', weights_only=True), strict=True)
        self.sparse.to(device).eval(); self.sparse.requires_grad_(False)
        self.ignored = {self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.unk_token_id}

    def encode(self, texts):
        dense = []; sparse = []
        for begin in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[begin:begin+self.batch_size], padding=True, truncation=False, return_tensors='pt').to(self.device)
            assert batch.input_ids.shape[1] <= 8192, 'Overlong BGE input; no silent truncation'
            with self.torch.inference_mode():
                hidden = self.model(**batch, return_dict=True).last_hidden_state
                dd = self.torch.nn.functional.normalize(hidden[:, 0], dim=-1)
                ww = self.torch.relu(self.sparse(hidden)).squeeze(-1)
            dense.extend(dd.cpu().tolist())
            sparse.extend(sparse_weights(ids, ws, self.ignored) for ids,ws in zip(batch.input_ids.cpu().tolist(), ww.cpu().tolist()))
        return dense, sparse
