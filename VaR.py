'''
import packages
'''
import sys
import json
import smtplib
import requests
import datetime
import pygsheets
import numpy as np
import pandas as pd
from scipy import stats
from pandas import DataFrame
from email.utils import formataddr
from sqlalchemy import create_engine
from email.mime.text import MIMEText

google_sheet_key = '1jJs1QVwIcbdZQk2WIJ7dQQRZDUQB11PumOVGEUQa2k8'
EOD_price = '343097694'
implied_vol_1d = '883443104'
implied_vol_1w = '2126506067'
symbols_conversion = '720521303'
gc = pygsheets.authorize()
spreadsheet = gc.open_by_key(google_sheet_key)


'''
weightage
'''


def _get_access_token_():
    # get oneZero access token
    url = "https://38.76.4.235:44300/api/token"
    headers = {'content-type': 'application/x-www-form-urlencoded',
               'grant_type': 'password', 'username': 'APITest',
               'password': 'cw12345..'}
    request = requests.post(url, data=headers, verify=False)
    data = request.json()
    return data['access_token']


def _get_lp_equity_(margin_account_number, access_token):
    # get free equity and margin of each lp
    url_equity = 'https://38.76.4.235:44300/api/rest/margin-account/' + \
        str(margin_account_number)
    request_equity = requests.get(
        url_equity, headers={'Authorization': 'Bearer ' + access_token}, verify=False)
    data_equity = json.loads(request_equity.text)
    free_equity = data_equity['freeMargin']
    margin = data_equity['margin']
    return free_equity, margin


def _get_weightage_(df_var_cov, df_weightage):
    # modify raw weightage according to index of var cov
    df_start = df_weightage[df_weightage['core_symbol'].str.find('USD') == 0]
    df_absent = df_weightage[df_weightage['core_symbol'].str.find('USD') == -1]
    df_end = df_weightage[df_weightage['core_symbol'].str.find('USD') == 4]
    df_start['dollarized_value'] = df_start['position']
    df_absent['dollarized_value'] = df_absent['position'] * \
        df_absent['exchange_rate']
    df_end['dollarized_value'] = df_end['position'] * df_end['price']
    df_weightage = pd.concat([df_start, df_absent, df_end])

    portfolio_value = df_weightage['dollarized_value'].sum()
    df_weightage['weightage'] = df_weightage['dollarized_value'] / \
        portfolio_value
    df_weightage = df_weightage.drop(
        ['core_symbol', 'exchange_rate', 'vol symbol', 'position', 'dollarized_value', 'price'], axis=1)
    df_weightage.set_index(['bbg symbol'], inplace=True)
    df_weightage = df_var_cov.join(df_weightage).iloc[:, -1:]
    print(df_weightage, '\n')
    print('portfolio value: ', portfolio_value, '\n')
    return df_weightage, portfolio_value


def _get_symbol_conversion_():
    # get symbol conversion from google drive
    df_symbols_conversion = DataFrame(spreadsheet.worksheet(
        property='id', value=symbols_conversion).get_all_records())
    return df_symbols_conversion


def _get_lp_position_(margin_account_number, access_token):
    # get raw weightage
    url_position = 'https://38.76.4.235:44300/api/rest/margin-account/' + \
        str(margin_account_number) + '/positions'
    request_position = requests.get(url_position, headers={
                                    'Authorization': 'Bearer ' + access_token}, verify=False)

    data_position = json.loads(request_position.text)['data']
    df_lp_positions = DataFrame(columns=['core_symbol', 'position', 'price'])
    for symbol_position in data_position:
        df_lp_positions.loc[len(df_lp_positions)] = [symbol_position['coreSymbol'].strip('v').strip('|').strip(
            '.'), float(symbol_position['position']), float(symbol_position['adapterPositions'][0]['marketPrice'])]
    df_weightage = pd.merge(_get_symbol_conversion_(),
                            df_lp_positions, on='core_symbol', how='left').fillna(0)
    return df_weightage


'''
price return
'''


def _get_price_var_cov_():
    # get price var cov
    df_EOD_price = DataFrame(spreadsheet.worksheet(
        property='id', value=EOD_price).get_all_records()).drop(['date'], axis=1)
    df_price_return = np.log(
        df_EOD_price.copy().pct_change() + 1).dropna().astype('float')
    mean = df_price_return.describe().loc[['mean']]
    means = DataFrame()
    for row in range(df_price_return.shape[0]):
        means = means.append(mean)
    means = means.reset_index(drop=True).astype('float')
    df_var_cov = (df_price_return - means).dropna().cov()
    print('var cov:\n', df_var_cov.head())
    return mean, df_price_return, df_var_cov


def _get_implied_var_cov_(df_var_cov, df_price_return, implied_vol):
    # get implied var cov
    df_implied_vol = DataFrame(spreadsheet.worksheet(
        property='id', value=implied_vol).get_all_records())
    df_implied_vol = pd.merge(_get_symbol_conversion_(),
                              df_implied_vol, on='vol symbol', how='left').fillna(0)
    df_implied_vol.drop(['core_symbol', 'exchange_rate',
                         'vol symbol'], axis=1, inplace=True)
    df_implied_vol.set_index(['bbg symbol'], inplace=True)
    print('df_implied_vol: \n', df_implied_vol)
    df_implied_vol = df_implied_vol / (np.sqrt(252) * 100)
    print(df_var_cov.join(df_implied_vol).iloc[:, -1:])
    var_cov_implied = np.matmul(df_var_cov.join(df_implied_vol).iloc[:, -1:].values, df_var_cov.join(
        df_implied_vol).iloc[:, -1:].T.values,                                out=None) * df_price_return.corr(method='pearson').as_matrix()
    print('implied var cov: \n', var_cov_implied)
    return var_cov_implied


'''
VaR result
'''


def _get_historical_(avg_return, std_dev, var_list, period, portfolio_value):
    # get normal historical var result
    confidence_lvls = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    var_list.append('normal')
    var_list.append(period)
    for confidence_lvl in confidence_lvls:
        var_list.append(np.abs(stats.norm.interval(
            confidence_lvl, avg_return, std_dev)[0] * portfolio_value))
    return var_list


def _get_monte_carlo_result_(avg_return, std_dev, var_list, period, portfolio_value):
    # get monte carlo result based on lp positions
    initial_index = 1.0
    i = 10000
    time_step = 1.0
    number_of_time_step = 50
    time_interval = time_step / number_of_time_step

    return_array = np.zeros((number_of_time_step + 1, i))
    return_array[0] = initial_index
    for time_step in range(1, number_of_time_step + 1):
        random_array = np.random.standard_normal(i)
        return_array[time_step] = return_array[time_step - 1] * np.exp(
            (avg_return - 0.5 * std_dev ** 2) *
            time_interval + std_dev * np.sqrt(time_interval) * random_array)
    rank_return = return_array[-1] - 1
    rank_return_df = pd.DataFrame(rank_return * portfolio_value)

    var_list.append('monte carlo')
    var_list.append(period)
    for confidence_lvl in [50, 40, 30, 20, 10, 5, 1]:
        var_list.append(np.abs(rank_return_df[rank_return_df < np.percentile(
            rank_return_df, confidence_lvl)].mean()[0]))
    return var_list


def _get_avg_dev_(mean, var_cov, df_weightage):
    # get avg and dev
    avg_return = float(np.matmul(mean.values, df_weightage.values)[0])
    dev = float(np.matmul(np.matmul(df_weightage.T.values,
                                    var_cov, out=None), df_weightage.values, out=None))
    std_dev = dev ** 0.5
    avg_return_5 = (avg_return + 1) ** 5 - 1
    std_dev_5 = (dev * 5) ** 0.5
    return avg_return, avg_return_5, std_dev, std_dev_5


def _get_var_result_(avg_return, avg_return_5, std_dev, std_dev_5, df_var, lp, portfolio_value, free_equity, margin, type):
    # consolidate var result
    var_list = [datetime.datetime.now().strftime(
        "%Y-%m-%d %H:00"), lp, free_equity, margin, type]
    df_var.loc[len(df_var)] = _get_historical_(
        avg_return, std_dev, var_list.copy(), 'one day', portfolio_value)
    df_var.loc[len(df_var)] = _get_historical_(
        avg_return_5, std_dev_5, var_list.copy(), 'one week', portfolio_value)
    df_var.loc[len(df_var)] = _get_monte_carlo_result_(
        avg_return, std_dev, var_list.copy(), 'one day', portfolio_value)
    df_var.loc[len(df_var)] = _get_monte_carlo_result_(
        avg_return_5, std_dev_5, var_list.copy(), 'one week', portfolio_value)


def _save_lp_var_(df_var):
    # create_engine说明：dialect[+driver]://user:password@host/dbname[?key=value..]
    df_lp_var_info = DataFrame(df_var)
    engine = create_engine(
        'postgresql://postgres:12345@localhost:5432/VaR')
    print(df_lp_var_info)
    df_lp_var_info.to_sql("VaR", engine,
                          index=False, if_exists='append')


def _send_alert_(lp, types):
    my_sender = 'caowei0127@gmail.com'
    my_pass = 'caoWEI940127'
    my_user = 'pro@blackwellglobal.com'
    mail_msg = """
    <p><a href="https://app.powerbi.com/groups/me/dashboards/ff8942ff-39d7-403e-8b9c-7e9d2151df4b">Open this link to view VaR</a></p>
    """

    msg = MIMEText(mail_msg, 'html', 'utf-8')
    msg['From'] = formataddr(["Cao Wei", my_sender])
    msg['To'] = formataddr(["Pro", my_user])
    msg['Subject'] = lp + ' VaR: ' + types + ' alerts!!'

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(my_sender, my_pass)
    server.sendmail(my_sender, [my_user, ], msg.as_string())
    server.quit()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    access_token = _get_access_token_()
    df_var = DataFrame(columns=['timestamp', 'lp', 'free_equity', 'margin', 'type', 'method',
                                'period', 'c50', 'c60', 'c70', 'c80', 'c90', 'c95', 'c99'])
    mean, df_price_return, df_var_cov = _get_price_var_cov_()
    var_cov_implied = _get_implied_var_cov_(
        df_var_cov, df_price_return, implied_vol_1d)
    mc_lp = {10: 'LMAX', 11: 'Divisa', 22: 'Vantage'}
    #mc_lp = {10: 'LMAX'}
    for margin_account_number in mc_lp.keys():
        df_weightage, portfolio_value = _get_weightage_(
            df_var_cov, _get_lp_position_(margin_account_number, access_token))
        free_equity, margin = _get_lp_equity_(
            margin_account_number, access_token)
        avg_return, avg_return_5, std_dev, std_dev_5 = _get_avg_dev_(
            mean, df_var_cov.T.values, df_weightage)
        _get_var_result_(avg_return, avg_return_5, std_dev, std_dev_5, df_var,
                         mc_lp[margin_account_number], portfolio_value, free_equity, margin, 'historical')
        avg_return_implied, avg_return_5_implied, std_dev_implied, std_dev_5_implied = _get_avg_dev_(
            mean, var_cov_implied, df_weightage)
        _get_var_result_(avg_return_implied, avg_return_5_implied, std_dev_implied, std_dev_5_implied, df_var,
                         mc_lp[margin_account_number], portfolio_value, free_equity, margin, 'implied')
    print(df_var.fillna(0))
    _save_lp_var_(df_var.fillna(0))


if __name__ == "__main__":
    #schedule.every().hour.do(main)
    main()


'''while True:
    schedule.run_pending()
    time.sleep(1)'''
