def process_mngr_skill(train_df, test_df):
    high_col = 'hcc_manager_id_target_high'
    medium_col = 'hcc_manager_id_target_medium'
    col = 'manager_skill'
    for df in [train_df, test_df]:
        df[col] = df[medium_col]+2*df[high_col]

    return train_df, test_df, [col]