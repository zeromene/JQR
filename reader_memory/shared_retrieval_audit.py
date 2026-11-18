"""Fixed common-retrieval diagnostics; official snippets are post-hoc only."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from .choice_protocol_audit import save
from .diagnostics import sha256
from .four_reader_probe import READERS, SharedCounter, load as load_probe, pack, serialize
from .independent_pack import normalize
from .native_choice_pilot import read
from .personamem2 import load_history
from .retrieval import BM25
from .schema import fingerprint
from .token_windows import token_windows

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/shared_retrieval_audit/v1'
OUTPUT = ROOT/'runs/shared_retrieval_audit_v1'
EMBEDDER = ROOT/'models/Qwen3-Embedding-0.6B'
POLICIES = ('legacy','clean_query','lexical','dense','hybrid','hybrid_neighbors')
BUDGETS = (1024,2048)
TASK = "Given a user's current message, retrieve historical conversation passages relevant to the user's preferences and constraints."
ORACLE_AUDIT = ROOT/'runs/four_reader_probe_v1/retrieval_source_audit.json'


def query_content(text):
    if not text.lstrip().startswith('{'):
        return text
    try:
        value = ast.literal_eval(text)
    except (SyntaxError,ValueError):
        try: value = json.loads(text)
        except json.JSONDecodeError as exc: raise ValueError('Malformed structured query') from exc
    if not isinstance(value,dict) or set(value)!={'role','content'} or value['role']!='user' or not isinstance(value['content'],str) or not value['content'].strip():
        raise ValueError('Unsupported structured query')
    return value['content']


def document(group):
    return '\n'.join(f"{s['role']}: {s['text']}" for s in group.metadata['spans'])


def rank_scores(scores):
    return sorted(range(len(scores)),key=lambda i:(-scores[i],i))


def rrf(first,second,k=60):
    if set(first)!=set(second) or len(set(first))!=len(first) or len(set(second))!=len(second):
        raise ValueError('RRF requires matching complete unique ranks')
    scores = {i:0.0 for i in first}
    for rank in (first,second):
        for place,index in enumerate(rank,1): scores[index]+=1/(k+place)
    return rank_scores([scores[i] for i in range(len(scores))]),scores


def neighbor_pack(order,groups,counter,budget):
    selected = []; indices = set(); events = []
    def add(index,seed,kind):
        if index<0 or index>=len(groups) or index in indices: return False
        proposal = [*selected,groups[index]]
        if not all(counter.count(serialize(proposal,a))<=budget for a in ('raw','compact','lead')): return False
        selected.append(groups[index]);indices.add(index)
        events.append({'chunk_id':groups[index].group_id,'seed_index':seed,'source_window_index':index,'reason':kind})
        return True
    for index in order[:24]:
        if index in indices:
            # An already retained neighbor may itself be a retrieved seed.
            continue
        if add(index,index,'retrieved-seed'):
            add(index-1,index,'preceding-source-window')
            add(index+1,index,'following-source-window')
    return {'text':serialize(selected,'raw'),'tokens_by_reader':counter.costs(serialize(selected,'raw')),
            'chunk_ids':[g.group_id for g in selected]},selected,events


class DenseEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDER,local_files_only=True,trust_remote_code=False,padding_side='left')
        self.model = AutoModel.from_pretrained(EMBEDDER,local_files_only=True,trust_remote_code=False,
                         dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        self.assets = {n:sha256(EMBEDDER/n) for n in ('source_manifest.json','config.json','tokenizer_config.json','tokenizer.json','model.safetensors')}

    def encode(self,texts,cache):
        signature = fingerprint({'assets':self.assets,'texts':texts,'pooling':'last-active-l2-float32','batch':16,'max_length':8192})
        cache = Path(cache); cache.mkdir(parents=True,exist_ok=True); path = cache/(signature+'.pt')
        if path.exists():
            value = self.torch.load(path,map_location='cpu',weights_only=True)
            if value.ndim!=2 or value.shape[0]!=len(texts) or not self.torch.isfinite(value).all():
                raise ValueError('Invalid embedding cache')
            return value,{'path':str(path.relative_to(ROOT).as_posix()),'sha256':sha256(path),'input_signature':signature,'reused':True}
        pieces = []
        for begin in range(0,len(texts),16):
            batch = self.tokenizer(texts[begin:begin+16],padding=True,truncation=False,return_tensors='pt').to('cuda')
            if batch['input_ids'].shape[-1]>8192: raise ValueError('Embedding input too long; refuse silent truncation')
            with self.torch.inference_mode():
                hidden = self.model(**batch,use_cache=False).last_hidden_state
                # Left padding: the final position is active for every example.
                if not self.torch.all(batch['attention_mask'][:,-1]==1): raise ValueError('Last-token pooling requires left padding')
                pooled = self.torch.nn.functional.normalize(hidden[:,-1].float(),p=2,dim=1).cpu()
            pieces.append(pooled)
        value = self.torch.cat(pieces)
        if not self.torch.isfinite(value).all(): raise ValueError('Nonfinite embeddings')
        temporary = path.with_suffix('.tmp'); self.torch.save(value,temporary);temporary.replace(path)
        return value,{'path':str(path.relative_to(ROOT).as_posix()),'sha256':sha256(path),'input_signature':signature,'reused':False}


def prepare(counters,encoder,study=STUDY,output=OUTPUT):
    study = Path(study);output = Path(output)
    if study.exists(): raise FileExistsError('Preserve frozen retrieval inputs')
    if set(counters)!=set(READERS): raise ValueError('Missing reader counter')
    original,queries = load_probe();common = SharedCounter(counters);histories = {};results = [];cache_records = []
    output.mkdir(parents=True,exist_ok=True)
    qtexts = [query_content(q['question']) for q in queries]
    query_vectors,qcache = encoder.encode([f'Instruct: {TASK}\nQuery: {text}' for text in qtexts],output/'embedding_cache')
    cache_records.append(qcache)
    for number,(q,clean) in enumerate(zip(queries,qtexts)):
        turns,audit = load_history(q)
        if any(normalize(t.content)==normalize(clean) for t in turns if t.role=='user'):
            raise ValueError('Actual current user content duplicates a history turn')
        if audit!=q['history_audit']: raise ValueError('History audit changed')
        if q['history_path'] not in histories:
            groups = token_windows(turns,common,limit=384)
            vectors,cache = encoder.encode([document(g) for g in groups],output/'embedding_cache')
            cache_records.append(cache);histories[q['history_path']] = groups,vectors
        groups,vectors = histories[q['history_path']]
        positions = {g.group_id:i for i,g in enumerate(groups)}
        old_ranking = BM25(groups).search(q['question'],top_k=len(groups))
        cleaned = BM25(groups).search(clean,top_k=len(groups))
        content_groups = [replace(g,variants={'raw':document(g)}) for g in groups]
        lexical = BM25(content_groups).search(clean,top_k=len(groups))
        dense_scores = (query_vectors[number]@vectors.T).tolist();dense_order = rank_scores(dense_scores)
        lexical_order = [positions[g.group_id] for g,_ in lexical]
        hybrid_order,hybrid_scores = rrf(lexical_order,dense_order)
        rankings = {'legacy':[(positions[g.group_id],s) for g,s in old_ranking],
                    'clean_query':[(positions[g.group_id],s) for g,s in cleaned],
                    'lexical':[(positions[g.group_id],s) for g,s in lexical],
                    'dense':[(i,dense_scores[i]) for i in dense_order],
                    'hybrid':[(i,hybrid_scores[i]) for i in hybrid_order]}
        branches = {}
        for policy in POLICIES:
            ranked = rankings['hybrid' if policy=='hybrid_neighbors' else policy]
            for budget in BUDGETS:
                if policy=='hybrid_neighbors':
                    item,chosen,events = neighbor_pack([i for i,_ in ranked],groups,common,budget)
                else:
                    packed,chosen = pack([(groups[i],s) for i,s in ranked[:24]],common,budget)
                    item = packed['raw'];events = [{'chunk_id':g.group_id,'reason':'retrieved-seed'} for g in chosen]
                if policy=='legacy' and item!=q['evidence'][f'raw:{budget}']:
                    raise ValueError('Legacy baseline does not exactly reproduce frozen evidence')
                if any(v>budget for v in item['tokens_by_reader'].values()): raise ValueError('Evidence budget exceeded')
                source_map = {t.source_id:t for t in turns}
                spans = [s for g in chosen for s in g.metadata['spans']]
                if len({g.group_id for g in chosen})!=len(chosen): raise ValueError('Duplicate selected window')
                for s in spans:
                    t = source_map[s['source_id']]
                    if s['text']!=t.content[s['start']:s['end']] or s['role']!=t.role or s['date']!=t.date:
                        raise ValueError('Selected source is not verbatim')
                branches[f'{policy}:{budget}'] = {**item,'spans':spans,'selection_events':events}
        results.append({k:q[k] for k in ('question_id','source_group','persona_id','history_path','question')}
                       | {'normalized_query':clean,'history_audit':audit,'branches':branches,
                          'rankings':{p:[{'chunk_id':groups[i].group_id,'source_window_index':i,'score':s} for i,s in ranked] for p,ranked in rankings.items()}})
        save(output/'preparation_status.json',{'status':'preparing','questions':len(results),'total':len(queries)})
        print('SHARED_RETRIEVAL_PREPARED',len(results),'/',len(queries),flush=True)
    study.mkdir(parents=True)
    save(study/'inputs.json',results)
    paths = ['research/SHARED_RETRIEVAL_AUDIT_PLAN.md','reader_memory/shared_retrieval_audit.py','reader_memory/choice_protocol_audit.py',
             'research/four_reader_probe/v1/manifest.json','research/four_reader_probe/v1/queries.json','runs/four_reader_probe_v1/retrieval_source_audit.json']
    m = {'version':'shared-retrieval-audit-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
         'file_hashes':{p:sha256(ROOT/p) for p in paths},'inputs_sha256':sha256(study/'inputs.json'),
         'questions':len(results),'source_groups':len(histories),'policies':list(POLICIES),'budgets':list(BUDGETS),
         'reader_assets':original['reader_assets'],'embedding_assets':encoder.assets,'embedding_caches':cache_records,
         'oracle_snippet_used_for_retrieval':False,'reserved_validation_used':False,'answer_inference_performed':False,
         'legacy_exactly_reproduced':True,'source_branch_checks':len(results)*len(POLICIES)*len(BUDGETS)}
    save(study/'manifest.json',m)
    print('SHARED_RETRIEVAL_FROZEN',m['source_branch_checks'],flush=True)
    return m


def load(study=STUDY):
    study = Path(study);m = read(study/'manifest.json')
    if any(sha256(ROOT/p)!=h for p,h in m['file_hashes'].items()) or sha256(study/'inputs.json')!=m['inputs_sha256']:
        raise ValueError('Frozen retrieval audit changed')
    load_probe()
    return m,read(study/'inputs.json')


def fully_covered(turn,spans):
    cursor = 0
    for start,end in sorted((s['start'],s['end']) for s in spans if s['source_id']==turn.source_id):
        if start<=cursor: cursor = max(cursor,end)
    return cursor==len(turn.content)


def report(study=STUDY,output=OUTPUT):
    # Inputs and ranks are now frozen. Only this phase reads oracle source IDs.
    m,inputs = load(study);output = Path(output);oracle = read(ORACLE_AUDIT)
    gold = {x['question_id']:x for x in oracle['rows']};rows = []
    original = {q['question_id']:q for q in load_probe()[1]}
    for q in inputs:
        turns,_ = load_history(original[q['question_id']]);turns = {t.source_id:t for t in turns}
        ref = gold[q['question_id']]; aligned = all(ref['matched_source_ids'])
        item = {'question_id':q['question_id'],'source_group':q['source_group'],'all_official_messages_aligned':aligned,'branches':{}}
        for key,branch in q['branches'].items():
            spans = branch['spans'];ids = {s['source_id'] for s in spans}
            matched_ids = [i for same in ref['matched_source_ids'] for i in same]
            item['branches'][key] = {
                'any_official_snippet_turn_touched':any(i in ids for i in matched_ids),
                'any_official_user_turn_touched':any(i in ids and turns[i].role=='user' for i in matched_ids),
                'all_official_snippet_messages_fully_retained':aligned and all(any(fully_covered(turns[i],spans) for i in same) for same in ref['matched_source_ids']),
                'selected_windows':len(branch['chunk_ids']),'selected_turns':len(ids),'retained_source_chars':sum(len(s['text']) for s in spans),
                'neighbor_windows':sum(e['reason']!='retrieved-seed' for e in branch['selection_events']),
                'tokens_by_reader':branch['tokens_by_reader']}
        rows.append(item)
    summaries = {}
    metrics = ('any_official_snippet_turn_touched','any_official_user_turn_touched','all_official_snippet_messages_fully_retained')
    for policy in POLICIES:
        summaries[policy] = {}
        for budget in BUDGETS:
            key = f'{policy}:{budget}';branches = [r['branches'][key] for r in rows]
            summaries[policy][str(budget)] = {'questions':len(rows),'fully_aligned_questions':sum(r['all_official_messages_aligned'] for r in rows),
                **{metric:sum(b[metric] for b in branches) for metric in metrics},
                'mean_selected_windows':sum(b['selected_windows'] for b in branches)/len(rows),
                'mean_neighbor_windows':sum(b['neighbor_windows'] for b in branches)/len(rows),
                'mean_retained_source_chars':sum(b['retained_source_chars'] for b in branches)/len(rows),
                'mean_tokens_by_reader':{r:sum(b['tokens_by_reader'][r] for b in branches)/len(rows) for r in READERS}}
    import numpy as np
    groups = sorted({r['source_group'] for r in rows});rng = np.random.default_rng(20261026)
    draws = rng.integers(0,len(groups),size=(20000,len(groups)))
    comparisons = []
    for candidate,base in (('clean_query','legacy'),('lexical','clean_query'),('dense','lexical'),('hybrid','lexical'),('hybrid_neighbors','hybrid')):
        for budget in BUDGETS:
            for metric in metrics:
                values = np.array([np.mean([int(r['branches'][f'{candidate}:{budget}'][metric])-int(r['branches'][f'{base}:{budget}'][metric])
                                   for r in rows if r['source_group']==g]) for g in groups])
                lo,hi = np.quantile(values[draws].mean(axis=1),[.025,.975])*100
                comparisons.append({'candidate':candidate,'baseline':base,'budget':budget,'metric':metric,
                    'difference_pp':float(values.mean()*100),'exploratory_group_bootstrap_95ci_pp':[float(lo),float(hi)]})
    value = {'status':'completed-shared-retrieval-audit','questions':len(rows),'source_groups':len(groups),'summaries':summaries,
             'paired_diagnostics':comparisons,'rows':rows,'source_branch_checks':m['source_branch_checks'],
             'oracle_used_only_after_input_freeze':True,'answer_inference_performed':False,'reserved_validation_used':False,
             'formal_benchmark_score':False,'bootstrap':{'draws':20000,'seed':20261026,'unit':'complete source group','multiple_comparison_correction':False}}
    save(output/'report.json',value)
    save(output/'status.json',{'status':value['status'],'questions':len(rows),'source_branch_checks':m['source_branch_checks']})
    print('SHARED_RETRIEVAL_REPORT',value['status'],len(rows),flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','report'));args = parser.parse_args()
    if args.command=='report': return report()
    from .__main__ import offline_environment
    from .tokenization import TokenCounter
    offline_environment()
    return prepare({r:TokenCounter(ROOT/'models'/r) for r in READERS},DenseEncoder())


if __name__=='__main__': main()
