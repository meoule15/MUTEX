"""Analyze reference-pool convergence and modality-dependent baseline effects."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def logmean(values, axes):
    values=np.asarray(values,dtype=np.float64)
    maximum=np.max(values,axis=axes,keepdims=True)
    result=maximum+np.log(np.mean(np.exp(values-maximum),axis=axes,keepdims=True))
    return np.squeeze(result,axis=axes)


def rank_agreement(scores, anchor, tasks):
    def sign(x): return np.where(np.abs(x)<1e-6,0,np.sign(x))
    gap=scores[:,0]-scores[:,1]; reference=anchor[:,0]-anchor[:,1]
    step=float(np.mean(sign(gap)==sign(reference)))
    task=float(np.mean([sign(np.mean(gap[tasks==t]))==sign(np.mean(reference[tasks==t])) for t in np.unique(tasks)]))
    return step,task


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert (root/'complete.json').exists()
    frames=json.loads((root/'frames.json').read_text())
    data=np.load(root/'likelihoods.npz')
    lp=data['logp']; blanks=data['blank_logp']
    assert lp.shape==(len(frames),2,100,3) and np.isfinite(lp).all() and np.isfinite(blanks).all()
    tasks=np.array([r['task_id'] for r in frames])
    own=np.stack([lp[i,:,task,:] for i,task in enumerate(tasks)])
    pool=manifest['reference_tasks']
    assert len(pool)==94 and not set(pool)&set(tasks)
    anchor_ref=logmean(lp[:,:,pool,:],(2,3))
    anchor=own[:,:,0]-anchor_ref
    blank_scores=own[:,:,0]-blanks
    gate=manifest['stability_gate']
    rows=[]; saved_refs={}
    for seed, permutation in manifest['pool_permutations'].items():
        assert set(permutation)==set(pool)
        for size in manifest['pool_sizes']:
            reference=logmean(lp[:,:,permutation[:size],:],(2,3))
            saved_refs[int(seed),size]=reference
            error=np.abs(reference-anchor_ref)
            step,task=rank_agreement(own[:,:,0]-reference,anchor,tasks)
            rows.append(dict(seed=int(seed),pool_size=size,reference_mae=float(error.mean()),
                             reference_p90_error=float(np.quantile(error,.9)),
                             gl_mae=float(error[:,0].mean()),vid_mae=float(error[:,1].mean()),
                             step_ranking_agreement=step,task_ranking_agreement=task,
                             passed=bool(error.mean()<=gate['mean_absolute_reference_logp_error_nats'] and
                                         step>=gate['step_ranking_agreement'] and task>=gate['task_ranking_agreement'])))
    summaries=[]
    for size in manifest['pool_sizes']:
        group=[r for r in rows if r['pool_size']==size]
        refs=np.stack([saved_refs[seed,size] for seed in range(3)])
        summaries.append(dict(pool_size=size,worst_reference_mae=max(r['reference_mae'] for r in group),
                              worst_step_ranking_agreement=min(r['step_ranking_agreement'] for r in group),
                              worst_task_ranking_agreement=min(r['task_ranking_agreement'] for r in group),
                              mean_seed_spread_nats=float((refs.max(0)-refs.min(0)).mean()),
                              all_seeds_pass=all(r['passed'] for r in group)))
    with (root/'pool_stability.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stable=next((r['pool_size'] for r in summaries if r['pool_size']<94 and r['all_seeds_pass']),None)
    blank_step,blank_task=rank_agreement(blank_scores,anchor,tasks)
    legacy=own[:,:,0]-logmean(lp[:,:,[6,7],0],(2,))
    legacy_step,legacy_task=rank_agreement(legacy,anchor,tasks)
    variant0_ref=logmean(lp[:,:,pool,0],(2,))
    variant0_step,variant0_task=rank_agreement(own[:,:,0]-variant0_ref,anchor,tasks)
    variant_checks=[]
    for variant in [1,2]:
        step,task=rank_agreement(own[:,:,variant]-anchor_ref,anchor,tasks)
        variant_checks.append(dict(variant=variant,step_ranking_agreement=step,task_ranking_agreement=task))
    all100_ref=logmean(lp,(2,3))
    shared_ref=logmean(anchor_ref,(1,))
    shared_scores=own[:,:,0]-shared_ref[:,None]
    # A common denominator cancels in a fixed-action modality comparison.
    assert np.allclose(shared_scores[:,0]-shared_scores[:,1],own[:,0,0]-own[:,1,0],atol=1e-8)
    shared_step,shared_task=rank_agreement(shared_scores,anchor,tasks)
    per_task=[]
    for task in np.unique(tasks):
        indices=tasks==task
        per_task.append(dict(task_id=int(task),blank_lifts=blank_scores[indices].mean(0).tolist(),
                             full_mixture_lifts=anchor[indices].mean(0).tolist(),
                             shared_reference_lifts=shared_scores[indices].mean(0).tolist(),
                             blank_reference_logp=blanks[indices].mean(0).tolist(),
                             mixture_reference_logp=anchor_ref[indices].mean(0).tolist()))
    report=dict(frames=len(frames),episodes=len(set((r['task_id'],r['episode']) for r in frames)),
                empirical_anchor_tasks=94,variants_per_task=3,smallest_passing_subset=stable,
                gate=gate,pool_stability=summaries,per_task=per_task,
                blank_vs_full={'step_agreement':blank_step,'task_agreement':blank_task,
                               'mean_reference_logp_difference_nats':float(np.abs(blanks-anchor_ref).mean())},
                legacy_two_task_vs_full={'step_agreement':legacy_step,'task_agreement':legacy_task},
                variant0_only_vs_all_variants={'step_agreement':variant0_step,'task_agreement':variant0_task},
                candidate_variant_sensitivity=variant_checks,
                all100_vs_excluded94_mae_nats=float(np.abs(all100_ref-anchor_ref).mean()),
                common_reference_vs_separate={'step_agreement':shared_step,'task_agreement':shared_task},
                limitations=['Uniform empirical task/variant prior is an assumption; it is not an identified p(specification|history).',
                             '94-task reference is an exact finite-pool anchor, not ground truth.',
                             'Three subset seeds, six seen tasks, two training demonstrations, prefixes only.',
                             'Step observations are correlated; metrics are descriptive, not significance tests.',
                             'Passing a convergence gate does not establish task-success prediction.'])
    if 'mc_logp' in data:
        mc=data['mc_logp']; own_mc=data['mc_conditioned']; null_mc=data['mc_blank']
        n=manifest['mc_samples']
        assert mc.shape==(len(frames),2,n,2,100,3) and np.isfinite(mc).all()
        assert np.isfinite(own_mc).all() and np.isfinite(null_mc).all()
        errors=[]
        for i,task in enumerate(tasks):
            for m in range(2):
                error=np.abs(own_mc[i,m]-mc[i,m,:,m,task,0]); errors.extend(error.tolist())
                assert np.allclose(own_mc[i,m],mc[i,m,:,m,task,0],atol=1e-3,rtol=1e-5)
        def expected_reference(ids):
            return np.stack([logmean(mc[:,m,:,m,:,:][:,:,ids,:],(2,3)) for m in range(2)],axis=1)
        full_mc_reference=expected_reference(pool)
        sample_lift=own_mc-full_mc_reference
        expected=sample_lift.mean(-1)
        se=sample_lift.std(-1,ddof=1)/np.sqrt(n)
        online_rows=[]
        for seed,perm in manifest['pool_permutations'].items():
            for size in manifest['pool_sizes']:
                reference=expected_reference(perm[:size])
                estimates=(own_mc-reference).mean(-1)
                step,task=rank_agreement(estimates,expected,tasks)
                online_rows.append(dict(seed=int(seed),pool_size=size,
                                   mean_absolute_expected_lift_error=float(np.abs(estimates-expected).mean()),
                                   step_ranking_agreement=step,task_ranking_agreement=task,
                                   passed=bool(np.abs(estimates-expected).mean()<=.5 and step>=.9 and task>=5/6)))
        null_reference=np.stack([null_mc[:,m,:,m] for m in range(2)],axis=1)
        blank_expected=(own_mc-null_reference).mean(-1)
        step,task=rank_agreement(blank_expected,expected,tasks)
        common_mc_reference=logmean(mc[:,:,:,:,pool,:],(3,4,5))
        common_expected=(own_mc-common_mc_reference).mean(-1)
        common_step,common_task=rank_agreement(common_expected,expected,tasks)
        report['online']=dict(samples_per_candidate=n,conditioned_batch_max_error=max(errors),
                              pool_stability=online_rows,blank_vs_full={'step_agreement':step,'task_agreement':task},
                              shared_vs_separate={'step_agreement':common_step,'task_agreement':common_task},
                              mean_sampling_standard_error=float(se.mean()),
                              uncertain_rank_fraction=float(np.mean(np.abs(expected[:,0]-expected[:,1])<=
                                                                    2*np.sqrt(se[:,0]**2+se[:,1]**2))))
        online_summary=[]
        for size in manifest['pool_sizes']:
            group=[r for r in online_rows if r['pool_size']==size]
            online_summary.append(dict(pool_size=size,
                 worst_expected_lift_error=max(r['mean_absolute_expected_lift_error'] for r in group),
                 worst_step_agreement=min(r['step_ranking_agreement'] for r in group),
                 worst_task_agreement=min(r['task_ranking_agreement'] for r in group),
                 all_seeds_pass=all(r['passed'] for r in group)))
        report['online']['pool_summary']=online_summary
        report['online']['smallest_passing_subset']=next((r['pool_size'] for r in online_summary
                                                        if r['pool_size']<94 and r['all_seeds_pass']),None)
        nested=[]
        for count in [32,64,128,256]:
            if count>n: continue
            draws=sample_lift[:,:,:count]
            estimate=draws.mean(-1)
            error=draws.std(-1,ddof=1)/np.sqrt(count)
            step,task=rank_agreement(estimate,expected,tasks)
            nested.append(dict(samples=count,mean_standard_error=float(error.mean()),
                               uncertain_rank_fraction=float(np.mean(np.abs(estimate[:,0]-estimate[:,1])<=
                                                           2*np.sqrt(error[:,0]**2+error[:,1]**2))),
                               step_agreement_with_all_samples=step,task_agreement_with_all_samples=task))
        report['online']['nested_sample_convergence']=nested
        if n>=64:
            first=sample_lift[:,:,:n//2].mean(-1); second=sample_lift[:,:,n//2:].mean(-1)
            step,task=rank_agreement(first,second,tasks)
            report['online']['independent_half_agreement']={'step_agreement':step,'task_agreement':task}
        if nested:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            counts=[r['samples'] for r in nested]
            axes[0].plot(counts,[r['mean_standard_error'] for r in nested],'-o')
            axes[0].set_ylabel('Mean Monte Carlo standard error (nats)')
            axes[1].plot(counts,[r['uncertain_rank_fraction'] for r in nested],'-o')
            axes[1].set_ylabel('Fraction of uncertainty-overlapping rankings'); axes[1].set_ylim(0,1)
            for ax in axes: ax.set_xlabel('Action samples per candidate')
            fig.tight_layout(); fig.savefig(root/'sampling_convergence.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in range(3):
        rs=[r for r in rows if r['seed']==seed]
        axes[0].plot([r['pool_size'] for r in rs],[r['reference_mae'] for r in rs],'-o',label=f'seed {seed}')
        axes[1].plot([r['pool_size'] for r in rs],[r['step_ranking_agreement'] for r in rs],'-o',label=f'seed {seed}')
    axes[0].axhline(.5,color='gray',linestyle='--'); axes[0].set_ylabel('Reference log-density MAE (nats)')
    axes[1].axhline(.9,color='gray',linestyle='--'); axes[1].set_ylabel('Step ranking agreement with 94-task anchor')
    axes[1].set_ylim(0,1.05)
    for ax in axes: ax.set_xlabel('Reference tasks (3 variants each)'); ax.legend()
    fig.suptitle('Empirical reference convergence; 94-task agreement is trivial by construction')
    fig.tight_layout(); fig.savefig(root/'pool_convergence.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(blank_scores[:,0]-blank_scores[:,1],anchor[:,0]-anchor[:,1],s=8,alpha=.4)
    ax.axhline(0,color='gray'); ax.axvline(0,color='gray')
    ax.set_xlabel('Language minus video lift: blank'); ax.set_ylabel('Language minus video lift: full mixture')
    fig.tight_layout(); fig.savefig(root/'baseline_ranking.png',dpi=160); plt.close(fig)
    (root/'report.json').write_text(json.dumps(report,indent=2))
    lines=['Reference distribution validation','',f"{len(frames)} matched actions across {report['episodes']} episodes.",
           f"Smallest tested subset passing all descriptive stability gates: {stable}",
           f"Blank vs full reference ranking agreement: {blank_step:.1%} of steps; {blank_task:.1%} of task means.",
           f"Historical two-task reference vs full: {legacy_step:.1%} of steps; {legacy_task:.1%} of task means.",
           f"Including target tasks changes log reference density by {report['all100_vs_excluded94_mae_nats']:.3f} nats on average.",
           '', 'Pool size / worst MAE / worst step agreement / worst task agreement:']
    lines += [f"{r['pool_size']} / {r['worst_reference_mae']:.3f} / {r['worst_step_ranking_agreement']:.1%} / {r['worst_task_ranking_agreement']:.1%}" for r in summaries]
    if 'online' in report:
        online=report['online']
        lines += ['',f"Expected-lift blank/full agreement: {online['blank_vs_full']['step_agreement']:.1%} of sampled states; "
                  f"{online['blank_vs_full']['task_agreement']:.1%} of task means.",
                  f"Smallest subset passing expected-lift stability gates across seeds: {online['smallest_passing_subset']}",
                  f"Sampling-uncertain modality rankings at {online['samples_per_candidate']} samples: {online['uncertain_rank_fraction']:.1%}."]
    lines += ['', 'Limits:']+report['limitations']
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='+')
    args=parser.parse_args()
    for directory in args.directories:
        report=analyze(directory)
        print(directory, 'frames',report['frames'],'smallest passing subset',report['smallest_passing_subset'])
