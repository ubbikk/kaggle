import subprocess

scripts = ['stacking_all.py',
           'stacking_base_and_hcc.py',
           'stacking_bid_prices_medians.py',
           'stacking_different_magic.py',
           'stacking_features_new.py',
           'stacking_naive_stats.py',
           'stacking_new_heu_all.py',
           'stacking_no_bid_hcc.py',
           'stacking_no_features.py',
           'stacking_no_lat_log.py',
           'stacking_no_listing_id.py',
           'stacking_no_magic.py',
           'stacking_no_mngr_count.py',
           'stacking_no_mngr_hcc.py',
           'stacking_no_mngr_hcc_and_avg_price.py',
           'stacking_no_mngr_medians.py',
           'stacking_no_nei_dummies.py',
           'stacking_no_nei_frequencies.py',
           'stacking_no_nei_median_ratios.py',
           'stacking_no_neis.py',
           'stacking_only_base_features.py',
           'stacking_random_forest.py',
           'stacking_street_avgs.py',
           'stacking_three_hcc.py',
           'stacking_weighted_price_ratio.py']

for s in scripts[13:16]:
    print '=================================='
    print 'running {}...'.format(s)
    print '=================================='
    subprocess.call(['python', '-u', s, '35.187.46.132'])

print '&&&&&&&&&&&&&&&&&&&&&&&&'
print 'Done'
print '&&&&&&&&&&&&&&&&&&&&&&&&'