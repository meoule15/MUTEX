"""Decompose fixed-action modality preferences using saved reference likelihoods."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pmi_reference import logmean


def preference(value):
    return np.where(np.abs(value)<1e-6,0,np.sign(value))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args(); root=args.directory
    manifest=json.loads((root/'manifest.json').read_text())
    frames=json.loads((root/'frames.json').read_text())
    data=np.load(root/'likelihoods.npz')
    lp=data['logp'].astype(np.float64); blank=data['blank_logp'].astype(np.float64)
    assert lp.shape==(len(frames),2,100,3) and np.isfinite(lp).all()
    pool=manifest['reference_tasks']
    assert not set(pool)&set(manifest['target_tasks'])
    mixture=logmean(lp[:,:,pool,:],(2,3))
    own=np.stack([lp[i,:,r['task_id'],:] for i,r in enumerate(frames)])
    reference_gap={'blank':blank[:,0]-blank[:,1],
                   'mixture':mixture[:,0]-mixture[:,1]}
    c=own[:,0,0]-own[:,1,0]
    gaps={key:c-value for key,value in reference_gap.items()}
    shared=logmean(mixture,(1,))
    common_gap=(own[:,0,0]-shared)-(own[:,1,0]-shared)
    assert np.allclose(common_gap,c,atol=1e-10)
    for key,ref in [('blank',blank),('mixture',mixture)]:
        direct=(own[:,0,0]-ref[:,0])-(own[:,1,0]-ref[:,1])
        assert np.allclose(direct,gaps[key],atol=1e-10)
    rows=[]
    for i,frame in enumerate(frames):
        row=dict(frame,language_conditioned_logp=float(own[i,0,0]),video_conditioned_logp=float(own[i,1,0]),
                 language_blank_logp=float(blank[i,0]),video_blank_logp=float(blank[i,1]),
                 language_mixture_logp=float(mixture[i,0]),video_mixture_logp=float(mixture[i,1]),
                 conditioned_gap=float(c[i]),blank_reference_gap=float(reference_gap['blank'][i]),
                 mixture_reference_gap=float(reference_gap['mixture'][i]),
                 blank_lift_gap=float(gaps['blank'][i]),mixture_lift_gap=float(gaps['mixture'][i]),
                 blank_reverses_conditioned=bool(preference(c[i])!=preference(gaps['blank'][i])),
                 mixture_reverses_conditioned=bool(preference(c[i])!=preference(gaps['mixture'][i])),
                 blank_mixture_ranking_flip=bool(preference(gaps['blank'][i])!=preference(gaps['mixture'][i])))
        row['action']=json.dumps(row['action'])
        rows.append(row)
    with (root/'term_decomposition.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tasks=np.array([f['task_id'] for f in frames])
    per_task=[]
    for task in np.unique(tasks):
        k=tasks==task
        per_task.append(dict(task_id=int(task),conditioned_gap=float(c[k].mean()),
                 blank_reference_gap=float(reference_gap['blank'][k].mean()),
                 mixture_reference_gap=float(reference_gap['mixture'][k].mean()),
                 blank_lift_gap=float(gaps['blank'][k].mean()),mixture_lift_gap=float(gaps['mixture'][k].mean())))
    comparisons={}
    for key in gaps:
        comparisons[key]=dict(mean_reference_gap=float(reference_gap[key].mean()),
             mean_lift_gap=float(gaps[key].mean()),
             reversals_from_conditioned=int(np.sum(preference(c)!=preference(gaps[key]))),
             reference_magnitude_exceeds_conditioned=int(np.sum(np.abs(reference_gap[key])>np.abs(c))),
             language_preferred=int(np.sum(gaps[key]>1e-6)))
    variants=[]
    for spec in range(3):
        conditioned=own[:,0,spec]-own[:,1,spec]
        b=conditioned-reference_gap['blank'];m=conditioned-reference_gap['mixture']
        variants.append(dict(spec_index=spec,blank_mixture_flips=int(np.sum(preference(b)!=preference(m))),
                             blank_reversals=int(np.sum(preference(b)!=preference(conditioned))),
                             mixture_reversals=int(np.sum(preference(m)!=preference(conditioned)))))
    report=dict(frames=len(frames),episodes=len(set((f['task_id'],f['episode']) for f in frames)),
                mean_conditioned_gap=float(c.mean()),conditioned_prefers_language=int(np.sum(c>1e-6)),
                blank_mixture_flips=int(np.sum(preference(gaps['blank'])!=preference(gaps['mixture']))),
                references=comparisons,per_task=per_task,candidate_variants=variants,
                blank_vs_mixture_mean_logp_deficit=(mixture-blank).mean(0).tolist(),
                checks={'decomposition_matches_direct_scores':True,'common_denominator_cancels':True},
                interpretation='Blank language reference assigns lower action densities than blank video reference on average, boosting language lift disproportionately.',
                limits=['Fixed demonstration actions, six seen tasks, two episodes each, sampled prefixes.',
                        'The mixture is an empirical prior, not an identified true marginal.',
                        'Denominator-induced reversals are an arithmetic attribution, not proof of which ranking is correct.',
                        'This diagnosis does not explain runtime sampling uncertainty or establish switch causality.'])
    (root/'term_diagnosis.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,['blank','mixture']):
        ax.scatter(c,reference_gap[key],s=10,alpha=.5)
        low=min(c.min(),reference_gap[key].min());high=max(c.max(),reference_gap[key].max())
        ax.plot([low,high],[low,high],'--',color='gray',label='equal terms: lift preference boundary')
        ax.axhline(0,color='gray',linewidth=.5);ax.axvline(0,color='gray',linewidth=.5)
        ax.set_xlabel('Conditioned logp: language − video');ax.set_ylabel('Reference logp: language − video')
        ax.set_title(key);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'term_decomposition.png',dpi=160);plt.close(fig)
    lines=['# Likelihood-term diagnosis','',
           f"Analyzed {len(frames)} fixed demonstration actions across 12 episodes. All gaps below are language minus video, in nats.",'',
           '| Term | Mean gap |','| --- | ---: |',f'| Conditioned likelihood | {c.mean():.3f} |',
           f"| Blank reference likelihood | {reference_gap['blank'].mean():.3f} |",
           f"| Mixture reference likelihood | {reference_gap['mixture'].mean():.3f} |",
           f"| Lift with blank reference | {gaps['blank'].mean():.3f} |",
           f"| Lift with mixture reference | {gaps['mixture'].mean():.3f} |",'',
           'The conditioned terms favor video on average. The blank reference reverses that average preference because its language density is much lower than its video density. The conditioned terms never change between reference comparisons.','',
           f"- Blank reference reverses the conditioned preference in **{comparisons['blank']['reversals_from_conditioned']}/{len(frames)}** actions.",
           f"- Mixture reference reverses it in **{comparisons['mixture']['reversals_from_conditioned']}/{len(frames)}** actions.",
           f"- Changing blank to mixture flips the ranking in **{report['blank_mixture_flips']}/{len(frames)}** actions.",
           f"- The blank reference gap exceeds the conditioned gap in magnitude in **{comparisons['blank']['reference_magnitude_exceeds_conditioned']}/{len(frames)}** actions.",'',
           '## Per-task means','',
           '| Task | Conditioned gap | Blank reference gap | Mixture reference gap | Blank lift gap | Mixture lift gap |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['task_id']} | {r['conditioned_gap']:.3f} | {r['blank_reference_gap']:.3f} | {r['mixture_reference_gap']:.3f} | {r['blank_lift_gap']:.3f} | {r['mixture_lift_gap']:.3f} |" for r in per_task]
    lines += ['', '## Conclusion','',
              'The blank reference introduces a strong language-favoring offset relative to the conditioned likelihood comparison. This explains a substantial part of the observed reference sensitivity. It does not prove that raw likelihood or the mixture ranking predicts success.', '',
              'Next, compare correct and mismatched specifications with a fixed empirical reference. That checks whether the remaining conditioned likelihood differences reflect useful task information.', '',
              '[Per-action terms](term_decomposition.csv) · [Decomposition plot](term_decomposition.png) · [Full numerical results](term_diagnosis.json)', '',
              '## Limits','']+['- '+v for v in report['limits']]
    (root/'TERM_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
