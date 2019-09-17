def get_gaintable(df,pred,label,bins=20,output=False):
    def get_percent(num):
        num=num*100
        return '%.2f%%'%num
    df=df[((df[label]==0)|(df[label]==1))]
    df=df[[pred,label]]
    df=df.sort_values(by=[pred],ascending=True)
    df['range']=range(len(df))
    df['cut']=pd.cut(df['range'],bins)
    df['total']=1
    total_bad_num=df[label].sum()
    total_good_num=len(df)-df[label].sum()
    min_score_df=df.groupby(['cut'])[pred].min().reset_index().rename(columns={pred:'min_score'})
    max_score_df=df.groupby(['cut'])[pred].max().reset_index().rename(columns={pred:'max_score'})
    score_df=min_score_df.merge(max_score_df,on=['cut'],how='left') 
    score_df['min_score']=np.round(score_df['min_score'],4) 
    score_df['max_score']=np.round(score_df['max_score'],4)
    score_df['score_range']=score_df[['min_score','max_score']].apply(lambda x:'{0}~{1}'.format(x[0],x[1]),axis=1)
    badnum_df=df.groupby(['cut'])[label].sum().reset_index().rename(columns={label:'bad_num'}) 
    num_df=df.groupby(['cut'])['total'].sum().reset_index()
    sample=df[['cut']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'bucket'})
    sample['bucket']=sample['bucket']+1
    sample=sample.merge(score_df[['cut','score_range']],on=['cut'],how='left')
    sample=sample.merge(num_df[['cut','total']],on=['cut'],how='left')
    sample=sample.merge(badnum_df[['cut','bad_num']],on=['cut'],how='left')
    sample['bad_rate']=(sample['bad_num']/sample['total'])
    sample['good_num']=sample['total']-sample['bad_num']
    sample=sample.sort_values(by=['bucket'],ascending=False)
    sample['cum_good']=sample['good_num'].cumsum()
    sample['gain']=(sample['bad_num']/total_bad_num).cumsum()
    sample['bucket']=(bins+1)-sample['bucket']
    sample['cumlift']=sample['gain']/((sample['bucket'])*1/bins)
    sample['cum_bad']=sample['bad_num'].cumsum()
    sample['cum_num']=sample['total'].cumsum()
    sample['cumbad_rate']=sample['cum_bad']/sample['cum_num']
    sample['ks']=np.abs((sample['cum_good']/total_good_num-sample['cum_bad']/total_bad_num))
    sample=sample[['bucket','total','bad_num','bad_rate','cumbad_rate','gain','ks','cumlift','score_range']]
    sample['bad_rate']=sample['bad_rate'].apply(get_percent)
    sample['cumbad_rate']=sample['cumbad_rate'].apply(get_percent)
    sample['gain']=sample['gain'].apply(get_percent)
    sample['cumlift']=sample['cumlift'].apply(lambda x:'%.2f'%x)
    sample['ks']=sample['ks'].apply(lambda x:'%.2f'%x)
    if output:
        sample.to_csv('gaintable.csv',index=None)
    return sample