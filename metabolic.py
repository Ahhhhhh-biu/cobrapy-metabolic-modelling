"""FBA/FVA and perturbation tutorial. Simulations are not experimental validation."""
import argparse, hashlib, json, platform
from pathlib import Path
import importlib.metadata as md
import numpy as np
import pandas as pd
from cobra.io import load_model, read_sbml_model, save_json_model
from cobra.flux_analysis import pfba, flux_variability_analysis
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients


def run(out_path, sbml=None):
    out=Path(out_path);out.mkdir(parents=True,exist_ok=False)
    model=read_sbml_model(sbml) if sbml else load_model('textbook')
    model.solver='glpk'
    objective={r.id:float(v) for r,v in linear_reaction_coefficients(model).items()}
    medium=dict(model.medium)
    for key in ['EX_glc__D_e','EX_o2_e']:
        if key not in model.reactions: raise ValueError('Adapt exchange IDs and protocol for this model')
    sol=model.optimize()
    if sol.status!='optimal' or sol.objective_value<=0:
        raise ValueError('Positive-growth optimal baseline required')
    baseline=float(sol.objective_value)
    tol=1e-7 # engineering numerical tolerance, not a biological threshold
    flux=sol.fluxes.reindex([r.id for r in model.reactions]).to_numpy()
    residual=float(np.max(np.abs(create_stoichiometric_matrix(model) @ flux)))
    bounds=max([max(r.lower_bound-v,v-r.upper_bound,0) for r,v in zip(model.reactions,flux)])
    assert residual<=tol and bounds<=tol
    par=pfba(model)
    biomass_pfba=float(sum(par.fluxes[r]*v for r,v in objective.items()))
    assert abs(biomass_pfba-baseline)<=tol*max(1,baseline)
    save_json_model(model,str(out/'model_snapshot.json'))
    sol.fluxes.to_csv(out/'baseline_fluxes.csv',header=['flux'])
    reactions=[x for x in ['PFK','PGI','PYK'] if x in model.reactions]
    if not reactions: reactions=[r.id for r in model.reactions[:3]]
    fva=flux_variability_analysis(model,reaction_list=reactions,fraction_of_optimum=.95,processes=1)
    assert np.all(fva.minimum <= fva.maximum+tol)
    fva.to_csv(out/'fva_95pct_optimum.csv')
    rows=[]
    for oxygen in [0.,5.,20.]:
        last=None
        for glucose in [1.,5.,10.]:
            with model:
                new=dict(medium);new['EX_glc__D_e']=glucose;new['EX_o2_e']=oxygen;model.medium=new
                result=model.optimize()
                value=float(result.objective_value) if result.status=='optimal' else None
                if last is not None and value is not None: assert value+tol>=last
                if value is not None: last=value
                rows.append({'glucose_uptake_cap':glucose,'oxygen_uptake_cap':oxygen,
                             'status':result.status,'predicted_growth':value})
    pd.DataFrame(rows).to_csv(out/'medium_sweep.csv',index=False)
    kos=[]
    for gene in sorted(model.genes,key=lambda g:g.id)[:5]:
        with model:
            gene.knock_out();result=model.optimize()
            value=float(result.objective_value) if result.status=='optimal' else None
            if value is not None: assert value<=baseline+tol
            kos.append({'gene':gene.id,'status':result.status,'predicted_growth':value,
                        'relative_growth':value/baseline if value is not None else None})
    pd.DataFrame(kos).to_csv(out/'five_gene_deletions.csv',index=False)
    restore=float(model.slim_optimize());assert abs(restore-baseline)<=tol
    meta={'status':'reference_model_demonstration','model':model.id,
          'scale':'core_tutorial' if not sbml else 'user_supplied_model_requires_scope_review',
          'reactions':len(model.reactions),'metabolites':len(model.metabolites),'genes':len(model.genes),
          'objective_coefficients':objective,'medium_uptake_caps':medium,
          'solver':'glpk','solver_configuration':{'timeout':model.solver.configuration.timeout, 'presolve':model.solver.configuration.presolve, 'feasibility_tolerance':model.solver.configuration.tolerances.feasibility, 'integrality_tolerance':model.solver.configuration.tolerances.integrality},
          'python':platform.python_version(),'versions':{k:md.version(k) for k in ['cobra','swiglpk','optlang','numpy','pandas']},
          'baseline_predicted_growth':baseline,'pfba_growth_objective':biomass_pfba,
          'max_steady_state_residual':residual,'max_bound_violation':float(bounds),
          'numerical_tolerance':tol,'state_restored':True,
          'snapshot_sha256':hashlib.sha256((out/'model_snapshot.json').read_bytes()).hexdigest(),
          'limitations':['Existing reconstruction, not a newly built metabolic network.',
          'Core tutorial does not establish genome-scale or human LARP1/mTOR modelling.',
          'FVA gives feasible flux ranges, not statistical confidence intervals.',
          'No flux sampling, omics-constrained modelling or ML integration has been implemented.',
          'No external growth or essentiality dataset has been validated.']}
    (out/'results.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ['model','reactions','baseline_predicted_growth','max_steady_state_residual']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--sbml');a=p.parse_args();run(a.out,a.sbml)
