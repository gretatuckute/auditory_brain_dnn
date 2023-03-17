"""
Functions pertaining diagnostics which was run in 2021/22.
"""
from plot_utils_AUD import *

#### DIAGNOSTICS FUNCTIONS ####

def loop_through_diagnostics(output_folders_paths, DIAGDIR, source_model, target, randnetw='False',
                             inv_std=True, inv_nan_constant_warning=False):
    """
    Loop through results output folders and investigate the 'ds' array which stores all regression-related values
    for each voxel and split.

    :param RESULTDIR: str
    :param output_folders: list of strings
    :param source_model: str
    :param randnetw: str, True or False
    :param inv_std: True if investigate standard deviation of predictions and neural data, else False
    :param inv_nan_constant_warning: True if investigate whether nans in r2 corrected coincide with constant warnings
                                    (not relevant if r2 corrected does not contain nan values)
    :return: Creates a diagnostics folder, stores csvs/plots
    """

    df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))
    df_str = 'ds.pkl'

    # Loop into folders and load ds:
    for i, f in tqdm(enumerate(output_folders_paths)):
        layer = f.split('SOURCE-')[1].split('_RAND')[0].split('-')[1:]
        if len(layer) == 1:
            layer = layer[0]
        else:
            layer = '-'.join(layer)

        ds = pd.read_pickle(join(f, df_str))

        if source_model == 'wav2vec' and layer == 'Logits':  # rename 'Logits' to Final
            layer = 'Final'

        ## ASSERT NANs IN ARRAYS OF INTEREST ##
        assert (np.sum(np.isnan(ds.r_test.values)) == 0)
        assert (np.sum(np.isnan(ds.r2_test.values)) == 0)
        assert (np.sum(np.isnan(ds.r2_train.values)) == 0)
        assert (np.sum(np.isnan(ds.r2_test_c.values)) == 0)

        ## INVESTIGATE STDS OF ACTUAL AND PREDICTED RESPONSES ##
        if inv_std:
            plot_stds(ds, source_model=source_model, layer=layer, target=target, save=DIAGDIR, randnetw=randnetw)

        ## CHECK WHETHER CONSTANT WARNINGS OCCURRED ##
        d = {}
        d['warning_constant_mean_COUNT'] = int(
            np.sum(ds.warning_constant_mean.values.flatten()))  # out of num voxels * splits, i.e. 7694*10=76940
        d['warning_constant_mean_PERC'] = np.round(
            (np.sum(ds.warning_constant_mean.values.flatten())) / len(ds.warning_constant_mean.values.flatten()) * 100,
            4)  # out of num voxels * splits, i.e. 7694*10=76940

        d['warning_constant_splits_COUNT'] = int(np.sum(
            ds.warning_constant_splits.values.flatten()))  # sum of how many times it occurred in a given CV split (the array can go up to 3)
        d['warning_constant_splits_PERC'] = np.round(
            (np.sum(ds.warning_constant_splits.values.flatten())) / (
                    len(ds.warning_constant_mean.values.flatten()) * 3) * 100, 4)
        # the warning for splits can be 1, 2 or 3 (for all estimation splits), but here the occurrence of it is just counted
        # therefore, there are 3 reliability estimators per split per voxel, i.e. 3*10*7694=230820

        reliability_vs_alpha(df_meta_roi, ds,
                             reliability_metric='kell_r_reliability',
                             save=join(DIAGDIR, f'kell_r_reliability-alphas_{layer}_{target}.png'))

        reliability_vs_alpha(df_meta_roi, ds,
                             reliability_metric='pearson_r_reliability',
                             save=join(DIAGDIR, f'pearson_r_reliability-alphas_{layer}_{target}.png'))

        if np.sum(ds.warning_constant_mean.values.flatten()) >= 1:
            reliability_vs_warning(df_meta_roi, ds=ds, target=target,
                                   reliability_metric='kell_r_reliability',
                                   warning_metric='warning_constant_mean',
                                   save=join(DIAGDIR, f'kell_r_reliability-warning_constant_mean_{layer}_{target}.png'))

            reliability_vs_warning(df_meta_roi, ds=ds, target=target,
                                   reliability_metric='pearson_r_reliability',
                                   warning_metric='warning_constant_mean',
                                   save=join(DIAGDIR,
                                             f'pearson_r_reliability-warning_constant_mean_{layer}_{target}.png'))

        if np.sum(ds.warning_constant_splits.values.flatten()) >= 1:
            reliability_vs_warning(df_meta_roi, ds=ds, target=target,
                                   reliability_metric='kell_r_reliability',
                                   warning_metric='warning_constant_splits',
                                   save=join(DIAGDIR,
                                             f'kell_r_reliability-warning_constant_splits_{layer}_{target}.png'))

            reliability_vs_warning(df_meta_roi, ds=ds, target=target,
                                   reliability_metric='pearson_r_reliability',
                                   warning_metric='warning_constant_splits',
                                   save=join(DIAGDIR,
                                             f'pearson_r_reliability-warning_constant_splits_{layer}_{target}.png'))

        ## CHECK WHETHER ALPHA WARNINGS OCCURRED ##
        # Upper hits denoted by 1
        d['warning_alphas_upper_COUNT'] = int(np.sum(ds.warning_alphas.values.flatten()[
                                                         ds.warning_alphas.values.flatten() == 1]))  # out of num voxels * splits, i.e. 7694*10=76940
        d['warning_alphas_upper_PERC'] = np.round(
            (np.sum(ds.warning_alphas.values.flatten()[ds.warning_alphas.values.flatten() == 1])) / len(
                ds.warning_alphas.values.flatten()) * 100,
            4)  # out of num voxels * splits, i.e. 7694*10=76940

        # Lower hits denoted by 2
        d['warning_alphas_lower_COUNT'] = int(np.sum(ds.warning_alphas.values.flatten()[
                                                         ds.warning_alphas.values.flatten() == 2]))  # out of num voxels * splits, i.e. 7694*10=76940
        d['warning_alphas_lower_PERC'] = np.round(
            (np.sum(ds.warning_alphas.values.flatten()[ds.warning_alphas.values.flatten() == 2])) / len(
                ds.warning_alphas.values.flatten()) * 100,
            4)  # out of num voxels * splits, i.e. 7694*10=76940

        ## Store how many times NaNs occurred ##
        d['nan_r_prior_zero_COUNT'] = int(np.sum(np.isnan(ds.r_prior_zero.values)))
        d['nan_r_prior_zero_PERC'] = np.round(
            int(np.sum(np.isnan(ds.r_prior_zero.values))) / (len(ds.r_prior_zero.values.flatten())) * 100, 4)

        d['nan_r2_test_c_COUNT'] = int(np.sum(np.isnan(ds.r2_test_c.values)))
        d['nan_r2_test_c_PERC'] = np.round(
            int(np.sum(np.isnan(ds.r2_test_c.values))) / (len(ds.r2_test_c.values.flatten())) * 100, 4)

        d['nan_r2_train_COUNT'] = int(
            np.sum(np.isnan(ds.r2_train.values)))  # just for the "main/mean" models, ie. out of vox*splits
        d['nan_r2_train_PERC'] = np.round(
            int(np.sum(np.isnan(ds.r2_train.values))) / (len(ds.r2_train.values.flatten())) * 100,
            4)  # just for the "main/mean" models, ie. out of vox*splits

        ## Check how many r2 c values exceeded 1 ##
        perc_r2_exceed1 = (np.sum(ds.r2_test_c.values.flatten() > 1) / (ds.r2_test_c.size) * 100)
        exceed_mask = ds.r2_test_c.values.flatten() > 1
        exceed_vals = ds.r2_test_c.values.flatten()[exceed_mask]

        d['exceed1_r2_test_c_PERC'] = np.round(perc_r2_exceed1, 4)
        d['exceed1_r2_test_c_MAX'] = np.round(exceed_vals.max(), 4)
        d['exceed1_r2_test_c_MEAN'] = np.round(np.mean(exceed_vals), 4)
        d['exceed1_r2_test_c_MEDIAN'] = np.round(np.median(exceed_vals), 4)

        if inv_nan_constant_warning:
            ## Check whether nan values and constant warnings occurred at same indices ##
            # (OBS, only relevant if there are actual nan values in r2 test)
            arr_warning_constant_mean = copy.deepcopy(ds.warning_constant_mean.values)
            arr_warning_constant_mean[arr_warning_constant_mean == 1] = np.nan  # set occurrences to nan

            arr_warning_constant_splits = copy.deepcopy(ds.warning_constant_splits.values)
            arr_warning_constant_splits[arr_warning_constant_splits >= 1] = np.nan  # set occurrences to nan

            idx_constant_mean = np.argwhere(np.isnan(arr_warning_constant_mean.flatten())).ravel()
            idx_constant_splits = np.argwhere(np.isnan(arr_warning_constant_splits.flatten())).ravel()

            idx_nan_r2_test_c = np.argwhere(np.isnan(ds.r2_test_c.values.flatten())).ravel()

            perc_overlap_nan_r2_test_c_constant_mean = np.sum(np.isin(idx_nan_r2_test_c, idx_constant_mean)) / len(
                idx_nan_r2_test_c) * 100
            perc_overlap_nan_r2_test_c_constant_splits = np.sum(np.isin(idx_nan_r2_test_c, idx_constant_splits)) / (len(
                idx_nan_r2_test_c)) * 100  # if this overlap is complete, then it means that unstable splits result in nans in r2 corrected

            d['overlap_nan_r2_test_c_constant_mean_PERC'] = np.round(perc_overlap_nan_r2_test_c_constant_mean, 4)
            d['overlap_nan_r2_test_c_constant_splits_PERC'] = np.round(perc_overlap_nan_r2_test_c_constant_splits, 4)

            ## For values that have nan r2, are the corresponding r values low?
            idx_non_nan_r2_test_c = np.delete(np.arange(len(ds.r2_test_c.values.flatten())), idx_nan_r2_test_c)

            ## This computation "just" takes the median, not the actual median over splits first, and then mean
            d['r_test_for_r2_test_c_nan_idx'] = np.round(np.median(ds.r_test.values.flatten()[idx_nan_r2_test_c]), 4)
            d['r_test_for_r2_test_c_non_nan_idx'] = np.round(
                np.median(ds.r_test.values.flatten()[idx_non_nan_r2_test_c]), 4)

        ## Get metric values where warning 1 or 0 for the constant mean warning ##
        metrics = ['r_prior_zero', 'r2_test_c', 'r2_test', 'r2_train', 'alphas']
        collapses = ['median']
        lst_stats = []
        for c in collapses:
            for m in metrics:
                d_stats = masked_stats(ds, mask='mean', metric=m, collapse=c)
                lst_stats.append(d_stats)

        d_stats_all = dict(pair for d in lst_stats for pair in d.items())

        ## Negative r-values ##
        # find negative r-values in the prior manipulation stored array
        neg_idx = [n < 0 for n in
                   ds.r_prior_zero.values.flatten()]  # the updated way of storing the original r values (not set to zero yet if r<0)
        num_neg_r = np.sum(neg_idx)

        r_test = ds.r_prior_zero.values.flatten()

        mean_neg_r = np.mean(
            r_test[neg_idx])  # find the mean r value (NOT r2, because that is all positive) for negative r indices
        median_neg_r = np.median(
            r_test[neg_idx])  # find the median r value (NOT r2, because that is all positive) for negative r indices
        neg_perc = num_neg_r * 100 / len(neg_idx)

        # check whether the neg r indices and constant warning indices overlap
        constant_warning_mean_idx = [n == 1 for n in ds.warning_constant_mean.values.flatten()]
        sum_not_equal_warning_idx = np.sum(~np.equal(neg_idx, constant_warning_mean_idx))
        sum_neg_r_and_constant_warning_idx = np.sum(constant_warning_mean_idx) + np.sum(
            neg_idx)  # where both arrays have True
        # idx_overlaps = sum_not_equal_idx - sum_idx # if this number is zero, then the

        d['neg_r_test_COUNT'] = int(num_neg_r)
        d['neg_r_test_MEAN'] = np.round(mean_neg_r, 4)
        d['neg_r_test_MEDIAN'] = np.round(median_neg_r, 4)
        d['neg_r_test_PERC'] = np.round(neg_perc, 4)
        d['neg_r_and_constant_warning_idx_COUNT'] = int(sum_neg_r_and_constant_warning_idx)
        d['not_equal_warning_idx_COUNT'] = int(sum_not_equal_warning_idx)

        # Merge dicts
        df = (pd.DataFrame.from_dict(data={**d, **d_stats_all}, orient='index'))
        df.to_csv(join(DIAGDIR, f'stats_{layer}_{target}.csv'))

        del ds  # clear


def loop_through_chunked_diagnostics(output_folders_paths, DIAGDIR, source_model, target, randnetw='False',
                                     inv_std=True, inv_nan_constant_warning=False):
    """
    Loop through results output folders and investigate the 'ds' array which stores all regression-related values
    for each voxel and split. Do not plot reliability vs alpha/warnings (already visualized in the other one)

    :param RESULTDIR: str
    :param output_folders: list of strings
    :param source_model: str
    :param randnetw: str, True or False
    :param inv_std: True if investigate standard deviation of predictions and neural data, else False
    :param inv_nan_constant_warning: True if investigate whether nans in r2 corrected coincide with constant warnings
                                    (not relevant if r2 corrected does not contain nan values)
    :return: Creates a diagnostics folder, stores csvs/plots
    """

    df_meta_roi = pd.read_pickle(join(DATADIR, 'neural', target, 'df_roi_meta.pkl'))
    df_str = 'ds.pkl'

    # Loop into folders and load ds:
    for i, f in tqdm(enumerate(output_folders_paths)):
        layer = f.split('SOURCE-')[1].split('_RAND')[0].split('-')[1:]
        if len(layer) == 1:
            layer = layer[0]
        else:
            layer = '-'.join(layer)

        if source_model == 'wav2vec' and layer == 'Logits':  # rename 'Logits' to Final
            layer = 'Final'

        # Run across chunks
        df_chunks = [i for i in os.listdir(f) if i.startswith(df_str[:-4] + '_')]
        if len(df_chunks) != 4:  # Assert that all chunks are there
            print(f'Missing chunks!')
            pdb.set_trace()

        for chunk_i, df_str_chunk in enumerate(df_chunks):
            ds = pd.read_pickle(join(f, df_str_chunk))

            ## ASSERT NANs IN ARRAYS OF INTEREST ##
            assert (np.sum(np.isnan(ds.r_test.values)) == 0)
            assert (np.sum(np.isnan(ds.r2_test.values)) == 0)
            assert (np.sum(np.isnan(ds.r2_train.values)) == 0)
            assert (np.sum(np.isnan(ds.r2_test_c.values)) == 0)

            ## INVESTIGATE STDS OF ACTUAL AND PREDICTED RESPONSES ##
            if inv_std:
                plot_stds(ds, source_model=source_model, layer=layer, target=f'{target}_chunk-{chunk_i}', save=DIAGDIR,
                          randnetw=randnetw)

            ## CHECK WHETHER CONSTANT WARNINGS OCCURRED ##
            d = {}
            d['warning_constant_mean_COUNT'] = int(
                np.sum(ds.warning_constant_mean.values.flatten()))  # out of num voxels * splits, i.e. 7694*10=76940
            d['warning_constant_mean_PERC'] = np.round(
                (np.sum(ds.warning_constant_mean.values.flatten())) / len(
                    ds.warning_constant_mean.values.flatten()) * 100,
                4)  # out of num voxels * splits, i.e. 7694*10=76940

            d['warning_constant_splits_COUNT'] = int(np.sum(
                ds.warning_constant_splits.values.flatten()))  # sum of how many times it occurred in a given CV split (the array can go up to 3)
            d['warning_constant_splits_PERC'] = np.round(
                (np.sum(ds.warning_constant_splits.values.flatten())) / (
                        len(ds.warning_constant_mean.values.flatten()) * 3) * 100, 4)
            # the warning for splits can be 1, 2 or 3 (for all estimation splits), but here the occurrence of it is just counted
            # therefore, there are 3 reliability estimators per split per voxel, i.e. 3*10*7694=230820

            ## CHECK WHETHER ALPHA WARNINGS OCCURRED ##
            # Upper hits denoted by 1
            d['warning_alphas_upper_COUNT'] = int(np.sum(ds.warning_alphas.values.flatten()[
                                                             ds.warning_alphas.values.flatten() == 1]))  # out of num voxels * splits, i.e. 7694*10=76940
            d['warning_alphas_upper_PERC'] = np.round(
                (np.sum(ds.warning_alphas.values.flatten()[ds.warning_alphas.values.flatten() == 1])) / len(
                    ds.warning_alphas.values.flatten()) * 100,
                4)  # out of num voxels * splits, i.e. 7694*10=76940

            # Lower hits denoted by 2
            d['warning_alphas_lower_COUNT'] = int(np.sum(ds.warning_alphas.values.flatten()[
                                                             ds.warning_alphas.values.flatten() == 2]))  # out of num voxels * splits, i.e. 7694*10=76940
            d['warning_alphas_lower_PERC'] = np.round(
                (np.sum(ds.warning_alphas.values.flatten()[ds.warning_alphas.values.flatten() == 2])) / len(
                    ds.warning_alphas.values.flatten()) * 100,
                4)  # out of num voxels * splits, i.e. 7694*10=76940

            ## Store how many times NaNs occurred ##
            d['nan_r_prior_zero_COUNT'] = int(np.sum(np.isnan(ds.r_prior_zero.values)))
            d['nan_r_prior_zero_PERC'] = np.round(
                int(np.sum(np.isnan(ds.r_prior_zero.values))) / (len(ds.r_prior_zero.values.flatten())) * 100, 4)

            d['nan_r2_test_c_COUNT'] = int(np.sum(np.isnan(ds.r2_test_c.values)))
            d['nan_r2_test_c_PERC'] = np.round(
                int(np.sum(np.isnan(ds.r2_test_c.values))) / (len(ds.r2_test_c.values.flatten())) * 100, 4)

            d['nan_r2_train_COUNT'] = int(
                np.sum(np.isnan(ds.r2_train.values)))  # just for the "main/mean" models, ie. out of vox*splits
            d['nan_r2_train_PERC'] = np.round(
                int(np.sum(np.isnan(ds.r2_train.values))) / (len(ds.r2_train.values.flatten())) * 100,
                4)  # just for the "main/mean" models, ie. out of vox*splits

            ## Check how many r2 c values exceeded 1 ##
            perc_r2_exceed1 = (np.sum(ds.r2_test_c.values.flatten() > 1) / (ds.r2_test_c.size) * 100)
            exceed_mask = ds.r2_test_c.values.flatten() > 1
            exceed_vals = ds.r2_test_c.values.flatten()[exceed_mask]

            d['exceed1_r2_test_c_PERC'] = np.round(perc_r2_exceed1, 4)
            d['exceed1_r2_test_c_MAX'] = np.round(exceed_vals.max(), 4)
            d['exceed1_r2_test_c_MEAN'] = np.round(np.mean(exceed_vals), 4)
            d['exceed1_r2_test_c_MEDIAN'] = np.round(np.median(exceed_vals), 4)

            if inv_nan_constant_warning:
                ## Check whether nan values and constant warnings occurred at same indices ##
                # (OBS, only relevant if there are actual nan values in r2 test)
                arr_warning_constant_mean = copy.deepcopy(ds.warning_constant_mean.values)
                arr_warning_constant_mean[arr_warning_constant_mean == 1] = np.nan  # set occurrences to nan

                arr_warning_constant_splits = copy.deepcopy(ds.warning_constant_splits.values)
                arr_warning_constant_splits[arr_warning_constant_splits >= 1] = np.nan  # set occurrences to nan

                idx_constant_mean = np.argwhere(np.isnan(arr_warning_constant_mean.flatten())).ravel()
                idx_constant_splits = np.argwhere(np.isnan(arr_warning_constant_splits.flatten())).ravel()

                idx_nan_r2_test_c = np.argwhere(np.isnan(ds.r2_test_c.values.flatten())).ravel()

                perc_overlap_nan_r2_test_c_constant_mean = np.sum(np.isin(idx_nan_r2_test_c, idx_constant_mean)) / len(
                    idx_nan_r2_test_c) * 100
                perc_overlap_nan_r2_test_c_constant_splits = np.sum(np.isin(idx_nan_r2_test_c, idx_constant_splits)) / (
                    len(
                        idx_nan_r2_test_c)) * 100  # if this overlap is complete, then it means that unstable splits result in nans in r2 corrected

                d['overlap_nan_r2_test_c_constant_mean_PERC'] = np.round(perc_overlap_nan_r2_test_c_constant_mean, 4)
                d['overlap_nan_r2_test_c_constant_splits_PERC'] = np.round(perc_overlap_nan_r2_test_c_constant_splits,
                                                                           4)

                ## For values that have nan r2, are the corresponding r values low?
                idx_non_nan_r2_test_c = np.delete(np.arange(len(ds.r2_test_c.values.flatten())), idx_nan_r2_test_c)

                ## This computation "just" takes the median, not the actual median over splits first, and then mean
                d['r_test_for_r2_test_c_nan_idx'] = np.round(np.median(ds.r_test.values.flatten()[idx_nan_r2_test_c]),
                                                             4)
                d['r_test_for_r2_test_c_non_nan_idx'] = np.round(
                    np.median(ds.r_test.values.flatten()[idx_non_nan_r2_test_c]), 4)

            ## Get metric values where warning 1 or 0 for the constant mean warning ##
            metrics = ['r_prior_zero', 'r2_test_c', 'r2_test', 'r2_train', 'alphas']
            collapses = ['median']
            lst_stats = []
            for c in collapses:
                for m in metrics:
                    d_stats = masked_stats(ds, mask='mean', metric=m, collapse=c)
                    lst_stats.append(d_stats)

            d_stats_all = dict(pair for d in lst_stats for pair in d.items())

            ## Negative r-values ##
            # find negative r-values in the prior manipulation stored array
            neg_idx = [n < 0 for n in
                       ds.r_prior_zero.values.flatten()]  # the updated way of storing the original r values (not set to zero yet if r<0)
            num_neg_r = np.sum(neg_idx)

            r_test = ds.r_prior_zero.values.flatten()

            mean_neg_r = np.mean(
                r_test[neg_idx])  # find the mean r value (NOT r2, because that is all positive) for negative r indices
            median_neg_r = np.median(
                r_test[
                    neg_idx])  # find the median r value (NOT r2, because that is all positive) for negative r indices
            neg_perc = num_neg_r * 100 / len(neg_idx)

            # check whether the neg r indices and constant warning indices overlap
            constant_warning_mean_idx = [n == 1 for n in ds.warning_constant_mean.values.flatten()]
            sum_not_equal_warning_idx = np.sum(~np.equal(neg_idx, constant_warning_mean_idx))
            sum_neg_r_and_constant_warning_idx = np.sum(constant_warning_mean_idx) + np.sum(
                neg_idx)  # where both arrays have True
            # idx_overlaps = sum_not_equal_idx - sum_idx # if this number is zero, then the

            d['neg_r_test_COUNT'] = int(num_neg_r)
            d['neg_r_test_MEAN'] = np.round(mean_neg_r, 4)
            d['neg_r_test_MEDIAN'] = np.round(median_neg_r, 4)
            d['neg_r_test_PERC'] = np.round(neg_perc, 4)
            d['neg_r_and_constant_warning_idx_COUNT'] = int(sum_neg_r_and_constant_warning_idx)
            d['not_equal_warning_idx_COUNT'] = int(sum_not_equal_warning_idx)

            # Merge dicts
            df = (pd.DataFrame.from_dict(data={**d, **d_stats_all}, orient='index'))
            df.to_csv(join(DIAGDIR, f'stats_{layer}_{target}_chunk-{chunk_i}.csv'))

            del ds  # clear


def loop_through_comp_diagnostics(output_folders_paths, DIAGDIR, source_model, target, randnetw='False', inv_std=True):
    """
    Loop through results output folders and investigate the 'ds' array which stores all regression-related values
    for each voxel and split.

    :param RESULTDIR: str
    :param output_folders: list of strings
    :param source_model: str
    :param randnetw: str, True or False
    :param inv_std: True if investigate standard deviation of predictions and neural data, else False
    :return: Creates a diagnostics folder, stores csvs/plots
    """

    df_str = 'ds.pkl'

    # Loop into folders and load ds:
    for i, f in tqdm(enumerate(output_folders_paths)):
        layer = f.split('SOURCE-')[1].split('_RAND')[0].split('-')[1:]
        if len(layer) == 1:
            layer = layer[0]
        else:
            layer = '-'.join(layer)
        ds = pd.read_pickle(join(f, df_str))

        if source_model == 'wav2vec' and layer == 'Logits':  # rename 'Logits' to Final
            layer = 'Final'

        ## ASSERT NANs IN ARRAYS OF INTEREST ##
        assert (np.sum(np.isnan(ds.r_test.values)) == 0)
        assert (np.sum(np.isnan(ds.r2_test.values)) == 0)
        assert (np.sum(np.isnan(ds.r2_train.values)) == 0)
        assert (np.sum(np.isnan(ds.r2_test_c.values)) == 0)

        ## INVESTIGATE STDS OF ACTUAL AND PREDICTED RESPONSES ##
        if inv_std:
            plot_stds(ds, source_model=source_model, layer=layer, target=target,
                      save=DIAGDIR, randnetw=randnetw, splits=False)

        ## CHECK WHETHER CONSTANT WARNINGS OCCURRED ##
        d = {}
        d['warning_constant_mean_COUNT'] = int(
            np.sum(ds.warning_constant_mean.values.flatten()))  # out of num voxels * splits, i.e.6*10=60
        d['warning_constant_mean_PERC'] = np.round(
            (np.sum(ds.warning_constant_mean.values.flatten())) / len(ds.warning_constant_mean.values.flatten()) * 100,
            4)  # out of num voxels * splits, i.e. 7694*10=76940

        ## CHECK WHETHER ALPHA WARNINGS OCCURRED ##
        # Upper hits denoted by 1
        d['warning_alphas_upper_COUNT'] = int(np.sum(ds.warning_alphas.values.flatten()[
                                                         ds.warning_alphas.values.flatten() == 1]))  # out of num voxels * splits, i.e. 7694*10=76940
        d['warning_alphas_upper_PERC'] = np.round(
            (np.sum(ds.warning_alphas.values.flatten()[ds.warning_alphas.values.flatten() == 1])) / len(
                ds.warning_alphas.values.flatten()) * 100,
            4)  # out of num voxels * splits, i.e. 7694*10=76940

        # Lower hits denoted by 2
        d['warning_alphas_lower_COUNT'] = int(np.sum(ds.warning_alphas.values.flatten()[
                                                         ds.warning_alphas.values.flatten() == 2]))  # out of num voxels * splits, i.e. 7694*10=76940
        d['warning_alphas_lower_PERC'] = np.round(
            (np.sum(ds.warning_alphas.values.flatten()[ds.warning_alphas.values.flatten() == 2])) / len(
                ds.warning_alphas.values.flatten()) * 100,
            4)  # out of num voxels * splits, i.e. 7694*10=76940

        ## Store how many times NaNs occurred ##
        d['nan_r_prior_zero_COUNT'] = int(np.sum(np.isnan(ds.r_prior_zero.values)))
        d['nan_r_prior_zero_PERC'] = np.round(
            int(np.sum(np.isnan(ds.r_prior_zero.values))) / (len(ds.r_prior_zero.values.flatten())) * 100, 4)

        d['nan_r2_test_c_COUNT'] = int(np.sum(np.isnan(ds.r2_test_c.values)))
        d['nan_r2_test_c_PERC'] = np.round(
            int(np.sum(np.isnan(ds.r2_test_c.values))) / (len(ds.r2_test_c.values.flatten())) * 100, 4)

        d['nan_r2_train_COUNT'] = int(
            np.sum(np.isnan(ds.r2_train.values)))  # just for the "main/mean" models, ie. out of vox*splits
        d['nan_r2_train_PERC'] = np.round(
            int(np.sum(np.isnan(ds.r2_train.values))) / (len(ds.r2_train.values.flatten())) * 100,
            4)  # just for the "main/mean" models, ie. out of vox*splits

        ## Get metric values where warning 1 or 0 for the constant mean warning ##
        metrics = ['r_prior_zero', 'r2_test', 'r2_train', 'alphas']
        collapses = ['median']
        lst_stats = []
        for c in collapses:
            for m in metrics:
                d_stats = masked_stats(ds, mask='mean', metric=m, collapse=c)
                lst_stats.append(d_stats)

        d_stats_all = dict(pair for d in lst_stats for pair in d.items())

        ## Negative r-values ##
        # find negative r-values in the prior manipulation stored array
        neg_idx = [n < 0 for n in
                   ds.r_prior_zero.values.flatten()]  # the updated way of storing the original r values (not set to zero yet if r<0)
        num_neg_r = np.sum(neg_idx)

        r_test = ds.r_prior_zero.values.flatten()

        mean_neg_r = np.mean(
            r_test[neg_idx])  # find the mean r value (NOT r2, because that is all positive) for negative r indices
        median_neg_r = np.median(
            r_test[neg_idx])  # find the median r value (NOT r2, because that is all positive) for negative r indices
        neg_perc = num_neg_r * 100 / len(neg_idx)

        # check whether the neg r indices and constant warning indices overlap
        constant_warning_mean_idx = [n == 1 for n in ds.warning_constant_mean.values.flatten()]
        sum_not_equal_warning_idx = np.sum(~np.equal(neg_idx, constant_warning_mean_idx))
        sum_neg_r_and_constant_warning_idx = np.sum(constant_warning_mean_idx) + np.sum(
            neg_idx)  # where both arrays have True
        # idx_overlaps = sum_not_equal_idx - sum_idx # if this number is zero, then the

        d['neg_r_test_COUNT'] = int(num_neg_r)
        d['neg_r_test_MEAN'] = np.round(mean_neg_r, 4)
        d['neg_r_test_MEDIAN'] = np.round(median_neg_r, 4)
        d['neg_r_test_PERC'] = np.round(neg_perc, 4)
        d['neg_r_and_constant_warning_idx_COUNT'] = int(sum_neg_r_and_constant_warning_idx)
        d['not_equal_warning_idx_COUNT'] = int(sum_not_equal_warning_idx)

        # Merge dicts
        df = (pd.DataFrame.from_dict(data={**d, **d_stats_all}, orient='index'))
        df.to_csv(join(DIAGDIR, f'stats_{layer}_{target}.csv'))

        del ds  # clear


def reliability_vs_warning(df_meta_roi, ds, target,
                           reliability_metric='kell_r_reliability', warning_metric='warning_constant_splits',
                           alpha=0.4, save=False):
    d = {'kell_r_reliability': 'Kell R reliability',
         'pearson_r_reliability': 'Pearson R reliability',
         'warning_constant_splits': 'Sum of warning across splits (max=30)',
         'warning_constant_mean': 'Sum of warning across splits (max=10)'}

    if warning_metric == 'warning_constant_splits':
        title_str = f'{d[reliability_metric]} versus warnings (models fitted on 1 session), {target}'
    else:
        title_str = f'{d[reliability_metric]} versus warnings (models fitted on all (3) sessions), {target}'

    plt.scatter(df_meta_roi[reliability_metric], ds[warning_metric].sum(axis=1).values, s=14, alpha=alpha)
    plt.ylabel(d[warning_metric])
    plt.xlabel(d[reliability_metric])
    plt.title(title_str, size='small')
    if save:
        plt.savefig(save)
    plt.show()


def reliability_vs_alpharange(df_meta_roi, problematic_vox, source_model, target, randnetw='False', save=None):
    """
    Plot problematic (i.e. voxels with alpha warnings) in red, other voxels in blue. Versus reliability.
    :param df_meta_roi:
    :param problematic_vox:
    :param source_model:
    :param randnetw:
    :param save:
    :return:
    """

    x = np.zeros(len(df_meta_roi))
    v = np.unique(problematic_vox)
    x[v] = 1
    colors = [[1, 0, 0] if y else [0, 0, 0.5] for y in x]

    plt.figure()
    plt.scatter(np.arange(len(df_meta_roi)), df_meta_roi['pearson_r_reliability'], c=colors, alpha=.3, s=10)
    plt.title(
        f'Pearson reliability with alpha out-of-range in voxels in red\n{source_model}, random network: {randnetw}, {target}',
        size='small')
    if save:
        plt.savefig(join(save, f'alpha_problematic_vox_pearson_r_reliability_{target}.png'))
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(np.arange(len(df_meta_roi)), df_meta_roi['kell_r_reliability'], c=colors, alpha=.3, s=10)
    plt.title(
        f'Kell reliability with alpha out-of-range in voxels in red\n{source_model}, random network: {randnetw}, {target}',
        size='small')
    if save:
        plt.savefig(join(save, f'alpha_problematic_vox_kell_r_reliability_{target}.png'))
    plt.tight_layout()
    plt.show()


def reliability_vs_alpha(df_meta_roi, ds,
                         reliability_metric='kell_r_reliability',
                         alpha=0.4, save=False):
    d = {'kell_r_reliability': 'Kell R reliability',
         'pearson_r_reliability': 'Pearson R reliability'}
    plt.figure(figsize=(10, 5))
    plt.scatter(df_meta_roi[reliability_metric], ds['alphas'].median(axis=1).values, s=14, alpha=alpha)
    plt.ylabel('Median of alphas across CV splits')
    plt.xlabel(f'{d[reliability_metric]}')
    plt.title(f'{d[reliability_metric]} versus alpha values', size='small')
    if save:
        plt.savefig(save)
    plt.tight_layout()
    plt.show()


def warning_across_voxels(ds):
    """NOT IN USE"""
    ## PLOTS OF WARNINGS ACROSS VOXELS MEAN/SPLIT ##
    plt.plot(ds['warning_constant_splits'].mean(axis=1), linewidth=0.5)
    plt.xlabel('Voxels')
    plt.ylabel('Occurrence of constant warning across splits')
    plt.title('Model fit on mean response across 2 sessions', size='small')
    plt.show()

    unique, counts = np.unique(ds['warning_constant_splits'].mean(axis=1), return_counts=True)
    count = dict(zip(unique, counts))
    print(count)

    plt.plot(ds['warning_constant_mean'].mean(axis=1), linewidth=0.5)
    plt.xlabel('Voxels')
    plt.ylabel('Occurrence of constant warning across splits')
    plt.title('Model fit on mean response across all (3) sessions', size='small')
    plt.show()

    unique, counts = np.unique(ds['warning_constant_mean'].mean(axis=1), return_counts=True)
    count = dict(zip(unique, counts))
    print(count)


def masked_stats(ds, mask='mean', metric='r2_test_c', collapse='median'):
    """Mask means 1 is a warning. Mask mean uses the warnings on the model fitted on all sessions (mean)
    Collapse refers to collapse over the CV splits
    OBS only works if the mask contains 1 as a flag"""

    d = {}
    m = (ds[f'warning_constant_{mask}']).astype(int)

    # print mask count
    unique, counts = np.unique(m, return_counts=True)
    count = dict(zip(unique, counts))
    print(f'Mask count: {count}')

    if mask == 'splits':  # vals in this array are 0, 1, 2, 3
        mask[mask > 1] = 1  # did not test this yet

    data = ds[metric].values
    data2 = copy.deepcopy(data)

    if collapse == 'median':
        data[
            m.values == 1] = np.nan  # set vals with 1 to nan, i.e get only normal ones (no warning, 0), i.e. no warning occurred
        data2[m.values == 0] = np.nan  # set the "good" indices to nan, only look at flagged indices (warning 1)

        n = np.nanmean(np.nanmedian(data, axis=1))
        w = np.nanmean(np.nanmedian(data2, axis=1))

    collapse_str = collapse.upper()
    d[f'{metric}_no_constant_warning_{mask}_{collapse_str}'] = np.round(n, 4)
    d[f'{metric}_constant_warning_{mask}_{collapse_str}'] = np.round(w, 4)

    print(f'{metric} for constant warning indices: {w:.3} (normal: {n:.3}). Collapse (over CV): {collapse}')

    return d


def plot_stds(ds, source_model, layer, target, save, randnetw='False', splits=True):
    """Visualize std of actual and predicted neural responses

    splits: if True, plot the std of models fitted on single rep
    """
    n_bins = 500
    alpha = 0.4
    lw = 0.5

    ## TEST ##
    ## normal plots
    plt.plot(np.median(ds.y_test_std_mean.values, axis=1), color='blue', alpha=alpha, linewidth=lw, label='y test')
    plt.plot(np.median(ds.y_pred_test_std_mean.values, axis=1), color='red', alpha=alpha, linewidth=lw,
             label='y test pred')
    plt.title(f'Std of y test versus y test predicted\n{source_model} {layer}, random network={randnetw}, {target}',
              size='small')
    plt.legend()
    if save:
        plt.savefig(join(save, f'y_mean_test_std_{layer}_{target}.png'))
    plt.show()

    ## histograms
    plt.hist(np.median(ds.y_test_std_mean.values, axis=1), bins=n_bins, color='blue', alpha=alpha, label='y test')
    plt.hist(np.median(ds.y_pred_test_std_mean.values, axis=1), bins=n_bins, color='red', alpha=alpha,
             label='y test pred')
    plt.title(f'Std of y test versus y test predicted\n{source_model} {layer}, random network={randnetw}, {target}',
              size='small')
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig(join(save, f'y_mean_test_std_hist_{layer}_{target}.png'))
    plt.show()

    ## TRAIN ##
    ## normal plots
    plt.plot(np.median(ds.y_train_std_mean.values, axis=1), color='blue', alpha=alpha, linewidth=lw, label='y train')
    plt.plot(np.median(ds.y_pred_train_std_mean.values, axis=1), color='red', alpha=alpha, linewidth=lw,
             label='y train pred')
    plt.title(f'Std of y train versus y train predicted\n{source_model} {layer}, random network={randnetw}, {target}',
              size='small')
    plt.legend()
    if save:
        plt.savefig(join(save, f'y_mean_train_std_{layer}_{target}.png'))
    plt.show()

    ## histograms
    plt.hist(np.median(ds.y_train_std_mean.values, axis=1), bins=n_bins, color='blue', alpha=alpha, label='y train')
    plt.hist(np.median(ds.y_pred_train_std_mean.values, axis=1), bins=n_bins, color='red', alpha=alpha,
             label='y train pred')
    plt.title(f'Std of y train versus y train predicted\n{source_model} {layer}, random network={randnetw}, {target}',
              size='small')
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig(join(save, f'y_mean_train_std_hist_{layer}_{target}.png'))
    plt.show()

    ## SPLITS ##
    if splits:
        for split in ['1', '2', '3']:
            plt.hist(np.median(ds.y_test_std_mean.values, axis=1), bins=n_bins, color='blue', alpha=alpha,
                     label='y test')
            plt.hist(np.median(ds[f'y_pred_std_split{split}'].values, axis=1), bins=n_bins, color='red', alpha=alpha,
                     label=f'y test pred, split {split}')
            plt.title(
                f'Std of y test versus y test predicted split {split}\n{source_model} {layer}, random network={randnetw}, {target}',
                size='small')
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(join(save, f'y_split{split}_test_std_hist_{layer}_{target}.png'))
            plt.show()


def check_alpha_ceiling(output_folders_paths, DIAGDIR, source_model, target, randnetw='False', save=True, ymax=50):
    """Check when there is a warning for alpha ceiling hit using log files.
    Plot the number of warnings across layers and write to csv.
    Write the median r2 corrected for alpha warnings versus non alpha warning voxels to csv.
    """
    layer_reindex = d_layer_reindex[source_model]
    upper_alpha_lim = '10000000000000000000000000000000000000000000000000'  # '100000000000000000000000000000'
    lower_alpha_lim = '1e-50'  # '1e-30'

    alpha_stats_all = []
    problematic_vox = []
    total = []
    upper = []
    lower = []
    layers = []
    for f_str in output_folders_paths:
        problematic_vox_layer = []  # get problematic voxels for each layer
        layer = f_str.split('SOURCE-')[1].split('_RAND')[0].split('-')[1:]
        if len(layer) == 1:
            layer = layer[0]
        else:
            layer = '-'.join(layer)

        # find log files
        log_files = []
        for file in os.listdir(f_str):
            if file.endswith('.log'):
                log_files.append(os.path.join(file))

        # get rid of hidden files
        log_files = [f for f in log_files if not f.startswith('._')]

        if len(log_files) > 1:
            print('More than one log')

        with open(join(f_str, log_files[0]), errors='replace') as log_file:
            log = log_file.readlines()

            warnings = []
            for line in log:
                if line.startswith('WARNING: BEST'):
                    try:
                        warnings.append(line)
                    except:
                        print(line)

            # find upper and lower boundary hits
            num_upper = 0
            num_lower = 0
            vox = []
            for w in warnings:
                x = w.split('ALPHA ')[1].split(' IS')[0]
                v = int(w.split('VOXEL ')[1].split(',')[0])
                vox.append(v)
                if x == upper_alpha_lim:
                    num_upper += 1
                if x == lower_alpha_lim:
                    num_lower += 1

            total.append(len(warnings))
            upper.append(num_upper)
            lower.append(num_lower)
            layers.append(layer)
            problematic_vox.append(vox)
            problematic_vox_layer.append(vox)

        # Figure out what the r2 was for problematic voxels
        ds = pd.read_pickle(join(f_str, 'ds.pkl'))
        problematic_vox_layer_u = np.unique(problematic_vox_layer)
        r2_vals = ds.r2_test_c.median(axis=1).values

        if problematic_vox_layer_u.size != 0:  # maybe there are no non-problematic vox
            non_problematic_vox = np.delete(np.arange(len(r2_vals)), problematic_vox_layer_u)
        else:
            non_problematic_vox = np.arange(len(r2_vals))

        ## Store separate alpha stats, with values for problematic versus nonproblematic voxels
        print(f'Layer: {layer}, problematic voxel median {np.nanmedian(r2_vals[problematic_vox_layer_u]):.3} '
              f'and non-problematic voxel mean {np.nanmedian(r2_vals[non_problematic_vox]):.3}')

        df_layerwise = pd.DataFrame({'r2_test_c_problematic_alpha_vox': np.nanmedian(r2_vals[problematic_vox_layer_u]),
                                     'r2_test_c_non_problematic_alpha_vox': np.nanmedian(r2_vals[non_problematic_vox])},
                                    index=[layer])
        alpha_stats_all.append(df_layerwise)

    ## ALL problematic voxels, across layers ##
    problematic_vox = np.asarray([item for sublist in problematic_vox for item in sublist])

    problematic_vox_u = np.unique(problematic_vox)
    print(
        f'Total number of problematic occurrences (across layers, voxels and splits): {len(problematic_vox)}, unique vox: {len(problematic_vox_u)}')

    df = pd.DataFrame({'total': total, 'upper': upper, 'lower': lower},
                      columns=['total', 'upper', 'lower'], index=layers)

    df = df.reindex(layer_reindex)

    # PLOT NUMBER OF OCCURRENCES #
    plt.plot(df['total'].values, label='Total')
    plt.plot(df['upper'].values, label='Upper')
    plt.plot(df['lower'].values, label='Lower')
    plt.xticks(np.arange(len(df)), df.index.values, rotation=75)
    plt.legend()
    plt.title(f'Number of alpha out-of-range occurrences\n{source_model}, random network: {randnetw}, {target}')
    plt.tight_layout()
    if save:
        plt.savefig(join(DIAGDIR, f'across-layers_alpha_warnings_COUNT_{target}.png'))
    plt.show()

    # PLOT PERCENTAGE OF OCCURRENCES #
    plt.plot(df['total'].values / (len(r2_vals) * 10) * 100, label='Total')
    plt.plot(df['upper'].values / (len(r2_vals) * 10) * 100, label='Upper')
    plt.plot(df['lower'].values / (len(r2_vals) * 10) * 100, label='Lower')
    plt.xticks(np.arange(len(df)), df.index.values, rotation=75)
    plt.legend()
    plt.ylim([0, ymax])
    plt.title(f'Percentage alpha out-of-range occurrences\n{source_model}, random network: {randnetw}, {target}')
    plt.tight_layout()
    if save:
        plt.savefig(join(DIAGDIR, f'across-layers_alpha_warnings_PERC_{target}.png'))
        plt.savefig(
            join(DIAGDIR_CENTRALIZED, f'across-layers_alpha_warnings_PERC_{source_model}_{randnetw}_{target}.png'))
    plt.show()

    df.to_csv(join(DIAGDIR, f'across-layers_alpha_warnings_{target}.csv'))

    # merge alpha stats across all layers
    df_layerwise_all = pd.concat(alpha_stats_all)
    df_layerwise_all = df_layerwise_all.reindex(layer_reindex)
    df_layerwise_all.to_csv(join(DIAGDIR, f'across-layers_stats_alpha_{target}.csv'))

    return vox


def plot_diagnostics(DIAGDIR, source_model, randnetw, target, val_of_interest='sum_warning_constant_mean',
                     ylim=None, save=False):
    """
     Plot diagnostics across all layers of a model. The function loop_through_diagnostics has to be run prior to this function

    :param RESULTDIR: str
    :param source_model: str
    :param randnetw: str, True or False
    :param val_of_interest: str, denoting the row in the stats csv file to plot
    :param randemb: str, True or False
    :param alpha_str: str, alpha range value
    :param ylim: lst, y limit for plot, else None
    :param save: bool, whether to save plot
    :return:
    """

    files = []
    for file in os.listdir(DIAGDIR):
        if file.startswith('stats'):
            if file.endswith(f'{target}.csv'):
                files.append(os.path.join(file))

    vals_lst = []
    layer_lst = []
    for f in files:
        s = pd.read_csv(join(DIAGDIR, f), names=['name', 'val'])
        v = s[s.name == val_of_interest].val.values
        vals_lst.append(v)
        layer_lst.append(f.split(f'_{target}')[0].split('stats_')[1])

    vals_lst = np.asarray(vals_lst).ravel()

    ## Reindex
    df = pd.DataFrame(vals_lst, index=layer_lst, columns=['val'])
    df = df.reindex(d_layer_reindex[source_model])

    plt.plot(df.val.values)
    plt.xticks(np.arange(len(d_layer_legend[source_model])), d_layer_legend[source_model], rotation=75)
    plt.ylabel(f'{val_of_interest}')
    if ylim:
        plt.ylim(ylim)
    plt.title(f'{source_model}, randnetw = {randnetw}, {target}', size='medium')
    plt.tight_layout()
    if save:
        plt.savefig(join(DIAGDIR, f'across-layers_{val_of_interest}_{target}.png'))
        plt.savefig(
            join(DIAGDIR_CENTRALIZED, f'across-layers_{val_of_interest}_{source_model}_{randnetw}_{target}.png'))
    plt.show()


def plot_two_diagnostics(DIAGDIR, source_model, randnetw, target, val_of_interest1='r_test_for_r2_test_c_nan_idx',
                         val_of_interest2='r_test_for_r2_test_c_non_nan_idx', ymax=None, save=False):
    """Plot diagnostics across all layers of a model. The function loop_through_diagnostics has to be run already"""

    files = []
    for file in os.listdir(DIAGDIR):
        if file.startswith('stats'):
            files.append(os.path.join(file))
    vals_lst1 = []
    vals_lst2 = []
    layer_lst = []
    for f in files:
        s = pd.read_csv(join(DIAGDIR, f), names=['name', 'val'])
        v1 = s[s.name == val_of_interest1].val.values
        v2 = s[s.name == val_of_interest2].val.values
        vals_lst1.append(v1)
        vals_lst2.append(v2)
        layer_lst.append(f.split('stats_')[1][:-4])

    vals_lst1 = np.asarray(vals_lst1).ravel()
    vals_lst2 = np.asarray(vals_lst2).ravel()

    ## Reindex
    df = pd.DataFrame({'val1': vals_lst1, 'val2': vals_lst2}, index=layer_lst)
    df = df.reindex(d_layer_reindex[source_model])

    plt.plot(df.val1.values, color='red', label=val_of_interest1)
    plt.plot(df.val2.values, color='green', label=val_of_interest2)
    plt.xticks(np.arange(len(d_layer_legend[source_model])), d_layer_legend[source_model], rotation=75)
    plt.legend()
    if ymax:
        plt.ylim([0, ymax])
    plt.title(f'{source_model}, randnetw = {randnetw}, {target}')
    plt.tight_layout()
    if save:
        plt.savefig(join(DIAGDIR, f'across-layers_[{val_of_interest1}]_[{val_of_interest2}]_{target}.png'))
    plt.show()


def r2_corrected_exceed1(output, source_model, target, randnetw=False, save=None):
    p = get_vox_by_layer_pivot(output, source_model, val_of_interest='median_r2_test_c')
    perc_r2_exceed1 = (np.sum(p.values > 1) / (p.size) * 100)
    print(
        f'Percentage of r2 corrected values that exceed 1 across all layers for median r2 test c: {perc_r2_exceed1:.4} %\n')

    # get mask of r2 corrected values > 1
    p_mask = p.values > 1

    # Test what the r2 (uncorrected) value was and what the r2 train value was
    p2 = get_vox_by_layer_pivot(output, source_model, val_of_interest='median_r2_test')
    p3 = get_vox_by_layer_pivot(output, source_model, val_of_interest='median_r2_train')

    print(
        f'The mean/median r2 test (uncorrected) for r2 corrected that exceeds one is {np.mean(p2.values[p_mask]):.4}/{np.median(p2.values[p_mask]):.4}'
        f' while for all other indices it is {np.mean(p2.values[~p_mask]):.4}/{np.median(p2.values[~p_mask]):.4}')  # obs, this just aggregates across all layers, no take
    # into account subjects etc
    print(
        f'The mean/median r2 train for r2 corrected that exceeds one is {np.mean(p3.values[p_mask]):.4}/{np.median(p3.values[p_mask]):.4}'
        f' while for all other indices it is {np.mean(p3.values[~p_mask]):.4}/{np.median(p3.values[~p_mask]):.4}')

    counts_to_plot = np.sum(p_mask, axis=0)

    plt.plot(counts_to_plot)
    plt.title(f'Median r2 test corrected > 1 occurrences \n{source_model}, random network: {randnetw}, {target}')
    plt.xticks(np.arange(len(p.columns.values)), p.columns.values, rotation=45)
    plt.xlabel('Count')
    plt.tight_layout()
    if save:
        plt.savefig(join(save, f'across-layers_median_r2_test_c_exceed1_COUNT_{target}.png'))
    plt.show()

    perc_to_plot = counts_to_plot / len(p) * 100  # divide by num vox

    plt.plot(perc_to_plot)
    plt.title(f'Median r2 test corrected > 1 occurrences \n{source_model}, random network: {randnetw}, {target}')
    plt.xticks(np.arange(len(p.columns.values)), p.columns.values, rotation=45)
    plt.xlabel('Percentage')
    plt.ylim([0, 50])
    plt.tight_layout()
    if save:
        plt.savefig(join(save, f'across-layers_median_r2_test_c_exceed1_PERC_{target}.png'))
        plt.savefig(join(DIAGDIR_CENTRALIZED,
                         f'across-layers_median_r2_test_c_exceed1_PERC_{source_model}_{randnetw}_{target}.png'))
    plt.show()
