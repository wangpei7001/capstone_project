import pandas as pd
import numpy as np
import os
import logging

def gathering_raw_data(path):
    files_list = [path + '/' + fname for fname in os.listdir(path)]
    
    list_frame = []
    
    logging.info('Gathering Data by Json....')

    for f in files_list:
        df_temp = pd.read_json(f)
        
        cols = set(df_temp.columns.tolist())
        
        if 'StreamID' in cols:
            df_temp.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in cols:
            df_temp.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in cols:
            df_temp.rename(columns={'total_price':'price'},inplace=True)
            
        list_frame.append(df_temp)
        
        
    df = pd.concat(list_frame)
    
    years, months, days = df['year'].values,df['month'].values,df['day'].values 
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    
    logging.info('Completed Step Gathering Data.')

    return df

def transform_timeseries(ds):
    
    logging.info('Transforming Data to TimeSeries....')

    start_month = '{}-{}'.format(ds['year'].values[0],str(ds['month'].values[0]).zfill(2))
    stop_month = '{}-{}'.format(ds['year'].values[-1],str(ds['month'].values[-1]).zfill(2))
    all_days_ts = np.arange(start_month, stop_month, dtype='datetime64[D]')
    dates = ds['invoice_date'].values.astype('datetime64[D]')
    
    list_ts = []
    
    for day in all_days_ts:
        count_purchases = np.where(dates==day)[0].size
        count_invoices_diff = np.unique(ds[dates==day]['invoice'].values).size
        count_streams_diff = np.unique(ds[dates==day]['stream_id'].values).size
        sum_views =  ds[dates==day]['times_viewed'].values.sum()
        sum_price_revenue = ds[dates==day]['price'].values.sum()
        
        obj_monted = {
                        'date': day,
                        'total_invoice': count_invoices_diff,
                        'purchase': count_purchases,
                        'total_streams': count_streams_diff,
                        'total_views': sum_views,
                        'revenue': sum_price_revenue
                     }
        
        list_ts.append(obj_monted)
        
    logging.info('Complete Time Series')
    return pd.DataFrame(data=list_ts)

def pipeline_gathering():
    logging.info('Start Pipeline Training')
    # 1. - Create a gathering data

    #   1.1 - Gathering data by json
    all_data = gathering_raw_data('cs-train')

    #   1.2 - Transform Data to TimeSeries and Busines Oportunity
    time_series_revenue = transform_timeseries(all_data)

    logging.info('Finish pipeline')
    return time_series_revenue.to_dict('records')
    



if __name__ == '__main__':
    pipeline_gathering()