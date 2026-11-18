"""Development-only learned marginal utility, with whole-persona exclusion."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .diagnostics import sha256
from .native_choice_pilot import READERS, read
from .schema import fingerprint, read_jsonl, write_json
from .update_memory import CHANGE, content_terms
from .window_utility import ROOT, OUTPUT as SOURCE, STUDY, VARIANTS, load as load_source, report as source_report

OUTPUT=ROOT/'runs/window_selector_v1'
MODES=('no_reader','reader_id')
ACTIONS=('without',*VARIANTS)
STRENGTH=10.


def feature(q,action,budget,reader,mode):
    if action not in VARIANTS:raise ValueError('Skip has known zero utility and no learned feature')
    groups=q['target_groups'];raw='\n'.join(g['variants']['raw'] for g in groups)
    text=q['evidence'][f'{action}:{budget}']['text'];background=q['plans'][str(budget)]['background']
    terms=content_terms(q['question']);raw_terms=content_terms(raw)
    selected_terms=content_terms(text[len(background):] if background else text)
    item=q['evidence'][f'{action}:{budget}'];base=q['evidence'][f'without:{budget}']
    structure=np.array([math.log1p(len(q['question']))/8, len(background)/max(1,len(raw)),
        base['tokens']/budget,(item['tokens']-base['tokens'])/budget,
        len(text[len(background):])/max(1,len(raw)),
        len(terms&raw_terms)/max(1,len(terms)),len(terms&selected_terms)/max(1,len(terms)),
        len(CHANGE.findall(raw))/max(1,len(raw.split())),len(groups)/2,
        float(q['link_trace']['linked']), float(q['link_trace'].get('after_anchor',False))])
    rep=np.array([float(action==a) for a in VARIANTS])
    shared=np.r_[1.,rep,np.outer(rep,structure).ravel()]
    if mode=='no_reader':return shared
    if mode!='reader_id':raise ValueError(mode)
    identity=np.array([float(reader==r) for r in READERS])
    return np.r_[shared,np.outer(identity,shared[1:]).ravel()]


class MarginalRidge:
    def __init__(self,mode,strength=STRENGTH):self.mode,self.strength=mode,strength

    def fit(self,queries,labels,training_personas):
        allowed=set(training_personas);qmap={q['question_id']:q for q in queries}
        selected=[u for u in labels if u['source_group'] in allowed]
        if not selected or any(qmap[u['question_id']]['source_group'] not in allowed for u in selected):
            raise ValueError('Invalid training source isolation')
        x=np.array([feature(qmap[u['question_id']],u['representation'],u['budget'],u['reader'],self.mode) for u in selected])
        y=np.array([u['delta_mean_accuracy'] for u in selected])
        self.center,self.scale=x.mean(axis=0),x.std(axis=0)
        self.center[0]=0;self.scale[0]=1;self.scale[self.scale<1e-12]=1
        z=(x-self.center)/self.scale;penalty=np.eye(z.shape[1])*self.strength;penalty[0,0]=0
        self.weights=np.linalg.solve(z.T@z+penalty,z.T@y)
        self.training_personas=sorted(allowed);self.training_question_ids=sorted({u['question_id'] for u in selected})
        return self

    def score(self,q,action,budget,reader):
        if action=='without':return 0.
        return float(((feature(q,action,budget,reader,self.mode)-self.center)/self.scale)@self.weights)

    def choose(self,q,budget,reader):
        values={a:self.score(q,a,budget,reader) for a in ACTIONS};maximum=max(values.values())
        tied=[a for a in ACTIONS if maximum-values[a]<1e-12]
        return min(tied,key=lambda a:(q['evidence'][f'{a}:{budget}']['tokens'],ACTIONS.index(a))),values

    def state(self):
        return {'mode':self.mode,'ridge_strength':self.strength,'training_personas':self.training_personas,
            'training_question_ids':self.training_question_ids,'center':self.center.tolist(),'scale':self.scale.tolist(),'weights':self.weights.tolist()}


def best_fixed(train, reader, budget, outcomes):
    scores={a:sum(outcomes[(reader,q['question_id'],budget,a)] for q in train) for a in ACTIONS}
    # Stable predeclared action order; no test outcomes or test costs.
    return max(ACTIONS,key=lambda a:scores[a])


def evaluate(queries,labels):
    outcomes={}
    for u in labels:
        outcomes[u['reader'],u['question_id'],u['budget'],u['representation']]=u['variant_correct']/4
        key=(u['reader'],u['question_id'],u['budget'],'without')
        if key in outcomes and outcomes[key]!=u['without_correct']/4:raise ValueError('Paired background outcomes differ')
        outcomes[key]=u['without_correct']/4
    persons=sorted({q['source_group'] for q in queries});predictions,models=[],[]
    for test_person in persons:
        train=[q for q in queries if q['source_group']!=test_person];test=[q for q in queries if q['source_group']==test_person]
        train_persons={q['source_group'] for q in train}
        fitted={mode:MarginalRidge(mode).fit(queries,labels,train_persons) for mode in MODES}
        for mode,model in fitted.items():models.append({'heldout_persona':test_person,**model.state()})
        for reader in READERS:
            for budget in (1024,2048):
                fixed=best_fixed(train,reader,budget,outcomes)
                for q in test:
                    selections={'always_raw':'bundle_raw','always_anchor':'anchor','training_best_fixed':fixed}
                    estimates={}
                    for mode,model in fitted.items():selections[mode],estimates[mode]=model.choose(q,budget,reader)
                    for method,action in selections.items():
                        predictions.append({'method':method,'reader':reader,'question_id':q['question_id'],
                            'source_group':test_person,'budget':budget,'selected_action':action,
                            'mean_accuracy_across_four_orders':outcomes[reader,q['question_id'],budget,action],
                            'evidence_tokens':q['evidence'][f'{action}:{budget}']['tokens'],
                            'predicted_utilities':estimates.get(method),'training_personas':sorted(train_persons)})
    return predictions,models


def run():
    r=source_report()
    if r['status']!='completed-window-utility':raise ValueError('Finish the utility acquisition first')
    manifest,queries=load_source();labels=list(read_jsonl(SOURCE/'utilities.train_only.jsonl'))
    settings={'version':'window-selector-v1','source_manifest_signature':fingerprint(manifest),
        'source_utilities_sha256':sha256(SOURCE/'utilities.train_only.jsonl'),
        'source_answers_sha256':sha256(SOURCE/'answers.jsonl'),'selector_code_sha256':sha256(__file__),
        'plan_sha256':sha256(ROOT/'research/WINDOW_SELECTOR_PLAN.md'),'ridge_strength':STRENGTH,
        'modes':list(MODES),'validation':'leave-one-exposed-persona-out development CV','formal_validation':False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'manifest.json').exists() and read(OUTPUT/'manifest.json')!=settings:
        raise ValueError('Preserve the existing selector run; use a new version')
    write_json(OUTPUT/'manifest.json',settings)
    predictions,models=evaluate(queries,labels)
    write_json(OUTPUT/'models.json',models)
    (OUTPUT/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions),encoding='utf-8')
    by_method={}
    for method in sorted({p['method'] for p in predictions}):
        selected=[p for p in predictions if p['method']==method]
        by_method[method]={'decisions':len(selected),'mean_accuracy':sum(p['mean_accuracy_across_four_orders'] for p in selected)/len(selected),
            'mean_evidence_tokens':sum(p['evidence_tokens'] for p in selected)/len(selected),
            'selected_actions':dict(Counter(p['selected_action'] for p in selected))}
    result={'status':'completed-development-window-selector','questions':len(queries),'source_personas':len({q['source_group'] for q in queries}),
        'by_method':by_method,'folds':len(models)//len(MODES),'capability_vector_efficacy_established':False,
        'unseen_reader_transfer_tested':False,'joint_multi_window_budget_optimization_tested':False,
        'formal_benchmark_score':False,'lme_heldout_used':False}
    write_json(OUTPUT/'report.json',result);print('WINDOW_SELECTOR_REPORT',json.dumps(result),flush=True)
    return result


if __name__=='__main__':run()
