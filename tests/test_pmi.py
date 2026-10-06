"""Run with python -m unittest discover -s tests -p test_pmi.py."""
import unittest
from collections import OrderedDict
import torch
from torch.distributions import Independent, Normal
from mutex.pmi import (action_lift, blank_specification, ActionLiftScorer,
                       expected_lift, SpecificationSelector)
from mutex.pmi import RandomSpecificationSelector
from mutex.pmi import log_mean_probability, batched_action_log_probs


def normal(mean):
    return Independent(Normal(torch.tensor([[float(mean)]]), torch.ones(1, 1)), 1)


class TestPMI(unittest.TestCase):
    def test_log_mixture_extreme_values_and_duplicates(self):
        values = torch.tensor([-10000., -10002.])
        result = log_mean_probability(values)
        self.assertTrue(torch.isfinite(result))
        torch.testing.assert_close(result, log_mean_probability(values.repeat(3)))
        with self.assertRaises(ValueError):
            log_mean_probability(torch.empty(0))

    def test_batched_reference_matches_single_and_preserves_history(self):
        class Policy:
            training = False
            steps = 1
            def _action_dist_from_latents(self, x, emb):
                return Independent(Normal(x[:, 0, :]+emb[:, 0, :], torch.ones(len(emb),1)),1)
        policy = Policy()
        x=torch.zeros(1,2,1); action=torch.ones(1,1); bank=torch.arange(7.).reshape(7,1,1)
        one=batched_action_log_probs(policy,x,bank,action,1)
        batch=batched_action_log_probs(policy,x,bank,action,3)
        torch.testing.assert_close(one,batch)
        samples=torch.stack([action, action+1])
        many=batched_action_log_probs(policy,x,bank,samples,3)
        self.assertEqual(many.shape,(2,7))
        torch.testing.assert_close(many[0],one)
        torch.testing.assert_close(many[1],batched_action_log_probs(policy,x,bank,action+1,1))
        self.assertEqual(policy.steps,1)
        policy.training=True
        with self.assertRaises(ValueError):
            batched_action_log_probs(policy,x,bank,action)

    def test_random_selector_reproducible_independent_rng(self):
        import random
        random.seed(14)
        state = random.getstate()
        first = RandomSpecificationSelector(['gl', 'vid'], 2, interval=10, seed=7)
        second = RandomSpecificationSelector(['gl', 'vid'], 2, interval=10, seed=7)
        for step in [1, 2, 11, 21, 31]:
            first.update(step); second.update(step)
            self.assertEqual(first.current, second.current)
        self.assertEqual(state, random.getstate())

    def test_selector_interval_batch_and_margin(self):
        selector = SpecificationSelector(['gl', 'vid'], 2, interval=10, margin=0.5)
        scores = {'gl': {'expected_lift': torch.tensor([1., 4.]),
                         'expected_lift_se': torch.zeros(2)},
                  'vid': {'expected_lift': torch.tensor([3., 2.]),
                          'expected_lift_se': torch.zeros(2)}}
        self.assertEqual(selector.update(1, scores), [True, False])
        self.assertEqual(selector.current, ['vid', 'gl'])
        scores['gl']['expected_lift'][0] = 8
        self.assertEqual(selector.update(2, scores), [False, False])
        self.assertEqual(selector.update(11, scores), [True, False])
        self.assertEqual(selector.current, ['gl', 'gl'])
        scores['vid']['expected_lift'][0] = 8.25
        self.assertEqual(selector.update(21, scores), [False, False])

    def test_selector_uncertainty_and_validation(self):
        selector = SpecificationSelector(['gl', 'vid'], 1)
        scores = {'gl': {'expected_lift': torch.tensor([1.]), 'expected_lift_se': torch.tensor([0.])},
                  'vid': {'expected_lift': torch.tensor([3.]), 'expected_lift_se': torch.tensor([2.])}}
        self.assertEqual(selector.update(1, scores), [False])
        self.assertEqual(selector.current, ['gl'])
        scores['vid']['expected_lift_se'][0] = float('nan')
        with self.assertRaises(ValueError):
            selector.update(11, scores)
        with self.assertRaises(ValueError):
            SpecificationSelector(['gl'], 1, interval=0)
        self.assertEqual(SpecificationSelector(['gl', 'vid'], 1).current, ['gl'])

    def test_expected_lift_and_entropy(self):
        torch.manual_seed(17)
        result = expected_lift(normal(0), [normal(2)], 20000)
        self.assertAlmostEqual(result['expected_lift'].item(), 2.0, delta=0.05)
        self.assertAlmostEqual(result['entropy'].item(), 1.41894, delta=0.05)
        self.assertGreater(result['expected_lift_se'].item(), 0)
        same = expected_lift(normal(0), [normal(0)], 32)
        self.assertEqual(same['expected_lift'].item(), 0)
        with self.assertRaises(ValueError):
            expected_lift(normal(0), [normal(0)], 1)

    def test_paired_references(self):
        class Policy:
            def get_action_dists(self, data, embeddings):
                return {k: normal(v) for k, v in embeddings.items()}
        scorer = ActionLiftScorer(Policy())
        results = scorer.score_step({}, torch.zeros(1, 1), {'gl': 0, 'vid': 2},
                                    {'gl_blank': 0, 'vid_blank': 2},
                                    {'gl': ['gl_blank'], 'vid': ['vid_blank']})
        self.assertTrue(all(r['lift'].item() == 0 for r in results.values()))
        with self.assertRaises(ValueError):
            scorer.score_step({}, torch.zeros(1, 1), {'gl': 0}, {'blank': 0}, {'gl': ['missing']})

    def test_identical_zero(self):
        d = normal(0)
        result = action_lift(d, [d], torch.zeros(1, 1))
        self.assertTrue(torch.equal(result['lift'], torch.zeros(1)))

    def test_mixture_probabilities(self):
        a = torch.zeros(1, 1)
        ds = [normal(0), normal(2)]
        result = action_lift(ds[0], ds, a)
        expected = torch.log(sum(d.log_prob(a).exp() for d in ds) / 2)
        torch.testing.assert_close(result['reference_logp'], expected)
        self.assertFalse(torch.allclose(expected, sum(d.log_prob(a) for d in ds)/2))

    def test_blank_preserves_mask_and_source(self):
        data = {'vid_spec': torch.ones(1, 3, 4), 'vid_spec_mask': torch.ones(1, 3)}
        blank = blank_specification(data)
        self.assertEqual(blank['vid_spec'].count_nonzero().item(), 0)
        self.assertIs(blank['vid_spec_mask'], data['vid_spec_mask'])
        self.assertEqual(data['vid_spec'].sum().item(), 12)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            action_lift(normal(0), [], torch.zeros(1, 1))
        with self.assertRaises(ValueError):
            action_lift(normal(0), [normal(0)], torch.full((1, 1), float('inf')))
        with self.assertRaises(ValueError):
            blank_specification({'gl_tokens': {}})

    def test_one_history_advance_and_reset(self):
        class Policy:
            def __init__(self): self.steps = 0
            def reset(self): self.steps = 0
            def get_action_dists(self, data, embeddings):
                self.steps += 1
                return OrderedDict((k, normal(v)) for k, v in embeddings.items())
        policy = Policy()
        scorer = ActionLiftScorer(policy)
        result = scorer.score_step({}, torch.zeros(1, 1), {'gl': 0, 'vid': 2}, {'blank': 0})
        self.assertEqual(policy.steps, 1)
        self.assertEqual(result['gl']['lift'].item(), 0)
        self.assertLess(result['vid']['lift'].item(), 0)
        scorer.reset()
        self.assertEqual(policy.steps, 0)


if __name__ == '__main__':
    unittest.main()
