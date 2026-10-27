from pathlib import Path
import json
O=Path('logs/rsl_rl/go2_velocity/acceptance_v2');arc=sorted(O.glob('observation_origin_archive_*'))[-1];n=0
for p in (arc/'runs').glob('*.json'):
 old=json.load(open(p));new=json.load(open(O/'runs'/p.name));assert old['groups']==new['groups'];assert old['raw_sha256']==new['raw_sha256']
 for ev,r in old['transients'].items():
  q=new['transients'][ev]
  assert r['per_environment']['settling_start_s']==q['per_environment']['settling_start_s']
  for key in ['eligible_envs','recovered','censored','recovered_without_reset','any_reset_envs','peak_vector_100ms_N','peak_mag_100ms_N','integral_vector_error_2s_Ns']:assert r['summary'][key]==q['summary'][key]
  for oldt,newt in zip(r['per_environment']['settling_start_s'],q['per_environment']['physical_recovery_start_s']):
   assert (oldt is None and newt is None) or abs(newt-oldt-.02)<1e-10
 n+=1
assert n>=11
r={'verified_runs':n,'raw_trajectories_identical':True,'all_absolute_relative_error_scores_identical':True,'event_peaks_and_censor_counts_identical':True,'physical_recovery_start_equals_legacy_start_plus20ms':True}
json.dump(r,open(O/'timing_update_verification.json','w'),indent=2);print(r)
