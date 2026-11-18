import itertools
import unittest
from reader_memory import proposal_retention_v1 as n


class RetentionTests(unittest.TestCase):
    def test_knapsack_matches_exhaustive_and_exact_lengths(self):
        items=[dict(cost=c,value=v) for c,v in [(3,4),(4,7),(2,3),(7,9),(1,1)]]
        dp=n.knapsack_table(items,12)
        brute={}
        for bits in itertools.product((0,1),repeat=len(items)):
            indices=tuple(i for i,b in enumerate(bits) if b)
            cost=sum(items[i]['cost'] for i in indices)
            if cost>12:continue
            value=sum(items[i]['value'] for i in indices)
            if cost not in brute or value>brute[cost][0] or (value==brute[cost][0] and indices<brute[cost][1]):
                brute[cost]=(value,indices)
        self.assertEqual(dp,[brute.get(i) for i in range(13)])

    def test_quote_mapping_expands_whole_sentences_and_checks_provenance(self):
        s=dict(paragraphs=[dict(source_id='0',text='Alpha. Beta.')])
        items=[dict(source='0',start=0,end=6),dict(source='0',start=7,end=12)]
        c=dict(source='0',start=3,end=9,text='ha. Be',roles=[0])
        p=dict(status='valid',roles=[{}],candidates=[c])
        self.assertEqual(n.proposal_core(s,items,p,'role'),([0,1],None))
        p['candidates'][0]['text']='invented'
        with self.assertRaises(AssertionError):n.proposal_core(s,items,p,'role')

    def test_incomplete_role_core_not_silently_used(self):
        p=dict(status='valid',roles=[{},{}],candidates=[dict(roles=[0])])
        self.assertEqual(n.proposal_core({},[],p,'role'),([], 'unusable_or_incomplete_proposal'))

    def test_greedy_preserves_core_and_skips_unaffordable(self):
        items=[dict(cost=7,score=10),dict(cost=4,score=9),dict(cost=2,score=1)]
        self.assertEqual(n.greedy(items,6,(2,)),(1,2))

if __name__=='__main__':unittest.main()
