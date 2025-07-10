from scipy import stats
from scipy.optimize import brentq
import numpy as np
from numpy import log, exp, sqrt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yfinance as yf
import datetime

# we download the data inside the '__main__' function but we have to convert expiry to time to maturity
def time_to_expiry(expiry_str):
    expiry_date = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    today = datetime.date.today()
    return (expiry_date - today).days / 365

# implement Black-Scholes model
def call_option_price(S, E, T, rf, sigma):
    # first we have to calculate d1 and d2 parameters
    d1 = (log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    # use the N(x) to calculate the price of the option
    return S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)

def put_option_price(S, E, T, rf, sigma):
    # first we have to calculate d1 and d2 parameters
    d1 = (log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    # use the N(x) to calculate the price of the option
    return -S*stats.norm.cdf(-d1)+E*exp(-rf*T)*stats.norm.cdf(-d2)

def implied_volatility(option_market_price, S, E, T, rf, option_type='call'):
    def objective(sigma):
        if option_type == 'call':
            return call_option_price(S, E, T, rf, sigma) - option_market_price
        else:
            return put_option_price(S, E, T, rf, sigma) - option_market_price
    try:
        iv = brentq(objective, 1e-6, 3)
        return iv
    except ValueError:
        return np.nan
    
# define butterfly arbitrage detector
def detect_butterfly_arbitrage(df):
    arbitrage_flags = []

    # loop through each expiry group
    for T, group in df.groupby('Expiry'):
        group_sorted = group.sort_values('Strike').reset_index(drop=True)

        for i in range(1, len(group_sorted) - 1):
            E1 = group_sorted.loc[i - 1, 'Strike']
            E2 = group_sorted.loc[i, 'Strike']
            E3 = group_sorted.loc[i + 1, 'Strike']
                
            C1 = group_sorted.loc[i - 1, 'Price']
            C2 = group_sorted.loc[i, 'Price']
            C3 = group_sorted.loc[i + 1, 'Price'] 

            butterfly_value = C1 - 2 * C2 + C3

            arbitrage_flags.append({'Expiry': T,
                                            'E1': E1,
                                            'E2': E2,
                                            'E3': E3,
                                            'C1': C1,
                                            'C2': C2,
                                            'C3': C3,
                                            'Butterfly Value': butterfly_value,
                                            'Arbitrage?': butterfly_value < 0
                                            })
                
    return pd.DataFrame(arbitrage_flags)

# define calendar arbitrage detector
def detect_calendar_arbitrage(df):
    arbitrage_flags = []

        # group by strike, then sort by expiry
    for E, group in df.groupby('Strike'):
        group_sorted = group.sort_values('Expiry').reset_index(drop=True)

        for i in range(len(group_sorted) - 1):
            T1 = group_sorted.loc[i, 'Expiry']
            T2 = group_sorted.loc[i + 1, 'Expiry']
            P1 = group_sorted.loc[i, 'Price']
            P2 = group_sorted.loc[i + 1, 'Price']

            calendar_diff = P2 - P1

            arbitrage_flags.append({
                                        'Strike': E,
                                        'T1': T1,
                                        'T2': T2,
                                        'P1': P1,
                                        'P2': P2,
                                        'Calendar Difference': calendar_diff,
                                        'Arbitrage?': calendar_diff < 0
                                    })
    
    return pd.DataFrame(arbitrage_flags)

# check call-put parity
def check_put_call_parity(df_calls, df_puts, E, rf):
    parity_violations = []
    for i, call in df_calls.iterrows():
        matching_puts = df_puts[
            (df_puts['Strike'] == call['Strike']) &
            (df_puts['Expiry'] == call['Expiry'])
        ]

        if not matching_puts.empty:
            put = matching_puts.iloc[0]
            lhs = call['Price'] - put['Price']
            rhs = E - call['Strike'] * np.exp(-rf * call['Expiry'])

            diff = abs(lhs - rhs)
            if diff > 1:  # $1 deviation threshold
                parity_violations.append({
                    'Strike': call['Strike'],
                    'Expiry': call['Expiry'],
                    'CallPrice': call['Price'],
                    'PutPrice': put['Price'],
                    'ParityDiff': diff
                })

    return pd.DataFrame(parity_violations)


if __name__ == '__main__':
    rf = 0.05
    ticker = yf.Ticker('SPY')
    expiries = ticker.options
    S0 = ticker.history(period='1d')['Close'].iloc[-1]
    
    # converting the dataset into a DataFrame
    data = []
    
    for expiry in expiries[:3]:  # use first 3 expiries (or more if needed)
        T = time_to_expiry(expiry)
        try:
            opt_chain = ticker.option_chain(expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
        except:
            continue
        
        # call option
        for _, row in calls.iterrows():
            strike = row['strike']
            price = row['lastPrice']
            iv = row['impliedVolatility']
            if price and iv:
                data.append({'Strike': strike, 'Expiry': T, 'Price': price, 'IV': iv, 'Option_type': 'Call'})
        
        # put option
        for _, row in calls.iterrows():
            strike = row['strike']
            price = row['lastPrice']
            iv = row['impliedVolatility']
            if price and iv:
                data.append({'Strike': strike, 'Expiry': T, 'Price': price, 'IV': iv, 'Option_type': 'Put'})
    
    df_prices = pd.DataFrame(data)
    calls_df = df_prices[df_prices['Option_type'] == 'Call'].copy()
    puts_df = df_prices[df_prices['Option_type'] == 'Put'].copy()
    calls_df['Option_type'] = 'Call'
    puts_df['Option_type'] = 'Put'
    parity_df = check_put_call_parity(calls_df, puts_df, S0, rf=0.05)
    parity_df['ParityDiff'] = abs(parity_df['CallPrice'] - parity_df['PutPrice'] - 
                                  (S0 - parity_df['Strike'] * np.exp(-rf * parity_df['Expiry'])))

    # run arbitrage detections
    butterfly_calls = detect_butterfly_arbitrage(calls_df)
    calendar_calls = detect_calendar_arbitrage(calls_df)

    butterfly_puts = detect_butterfly_arbitrage(puts_df)
    calendar_puts = detect_calendar_arbitrage(puts_df)
    
    # filter violations and junk results
    butterfly_violations = pd.concat([
        butterfly_calls[butterfly_calls['Arbitrage?'] == True],
        butterfly_puts[butterfly_puts['Arbitrage?'] == True]
    ], ignore_index=True)

    calendar_violations = pd.concat([
        calendar_calls[calendar_calls['Arbitrage?'] == True],
        calendar_puts[calendar_puts['Arbitrage?'] == True]
    ], ignore_index=True)

    parity_df = parity_df[parity_df['CallPrice'] > 0.5]
    parity_df = parity_df[parity_df['PutPrice'] > 0.5]
    parity_df = parity_df[parity_df['Expiry'] > 0.01]

    # print the results
    print("\n Butterfly Arbitrage Violations:")
    print(butterfly_violations)

    print("\n Calendar Arbitrage Violations:")
    print(calendar_violations)

    print("Put-Call Parity Violations:\n", parity_df)

    # plot volatility surface
    iv_grid = df_prices.pivot_table(index='Expiry', columns='Strike', values='IV')
    X, Y = np.meshgrid(iv_grid.columns.values, iv_grid.index.values)  # Strike and Expiry
    Z = iv_grid.values # IV values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.9)

    ax.set_title('Implied Volatility Surface')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expiry (Years)')
    ax.set_zlabel('Implied Volatility')

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

    # export to csv
    butterfly_violations.to_csv('butterfly_violations.csv', index=False)
    calendar_violations.to_csv('calendar_violations.csv', index=False)
    parity_df.to_csv('parity_df.csv', index=True)