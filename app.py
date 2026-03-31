import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/analyse', methods=['POST'])
def analyse():
    data = request.get_json()
    tickers = data.get('tickers', [])
    weights_raw = data.get('weights', [])
    period = data.get('period', '1y')

    if not tickers or not weights_raw or len(tickers) != len(weights_raw):
        return jsonify({'error': 'Invalid input'}), 400

    w = pd.Series(weights_raw, index=tickers)
    benchmark = '^GSPC'
    all_tickers = list(set(tickers + [benchmark]))

    try:
        df = yf.Tickers(all_tickers).history(period=period, auto_adjust=True)['Close']
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.tz_localize(None)
        df = df.dropna(how='all')

        for t in all_tickers:
            if t not in df.columns:
                return jsonify({'error': f'No data found for {t}'}), 400

        df = df.ffill().dropna()

        daily_returns = df.pct_change().dropna()
        portfolio_return = (daily_returns[tickers] * w).sum(axis=1)
        benchmark_return = daily_returns[benchmark]

        common_index = portfolio_return.index.intersection(benchmark_return.index)
        portfolio_return = portfolio_return[common_index]
        benchmark_return = benchmark_return[common_index]

        freq = 52 if period in ['5y', 'max'] else 252

        ann_vol = portfolio_return.std() * np.sqrt(freq)
        ann_ret = (1 + portfolio_return).prod() ** (freq / len(portfolio_return)) - 1 if len(portfolio_return) > 0 else 0
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

        var_b = benchmark_return.var()
        beta = portfolio_return.cov(benchmark_return) / var_b if var_b != 0 else 0

        cumulative = (1 + portfolio_return).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_drawdown = float(drawdown.min())

        def make_layout(title):
            return go.Layout(
                template='plotly_dark',
                title=dict(text=title, font=dict(color='white', size=14)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=11),
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(font=dict(size=10)),
                xaxis=dict(gridcolor='rgba(255,255,255,0.07)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.07)')
            )

        def to_html(fig):
            return pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                config={'displayModeBar': False, 'scrollZoom': True}
            )

        df100 = df[tickers] / df[tickers].iloc[0] * 100
        port100 = (portfolio_return + 1).cumprod() * 100

        fig1 = go.Figure(layout=make_layout('Basis 100'))
        for s in tickers:
            fig1.add_scatter(x=df100.index.astype(str).tolist(),
                             y=df100[s].tolist(), name=s, opacity=0.6)
        fig1.add_scatter(x=port100.index.astype(str).tolist(),
                         y=port100.tolist(), name='Portfolio',
                         line=dict(width=4, color='white'))

        fig2 = go.Figure(layout=make_layout('Daily Returns'))
        fig2.add_scatter(x=portfolio_return.index.astype(str).tolist(),
                         y=portfolio_return.tolist(), name='Portfolio')

        roll_vol = portfolio_return.rolling(20).std() * np.sqrt(freq)
        fig3 = go.Figure(layout=make_layout('Rolling Volatility'))
        fig3.add_scatter(x=roll_vol.index.astype(str).tolist(),
                         y=roll_vol.tolist(), name='Portfolio',
                         line=dict(width=3))

        fig4 = go.Figure(layout=make_layout('Weights'))
        fig4.add_pie(labels=list(w.index), values=list(w.values),
                     hole=0.5, textinfo='label+percent')

        fig5 = go.Figure(layout=make_layout('Drawdown (%)'))
        fig5.add_scatter(x=drawdown.index.astype(str).tolist(),
                         y=(drawdown * 100).tolist(), name='Drawdown',
                         fill='tozeroy', line=dict(color='red'))

        return jsonify({
            'ann_ret':      round(ann_ret * 100, 2),
            'ann_vol':      round(ann_vol * 100, 2),
            'sharpe':       round(float(sharpe), 4),
            'beta':         round(float(beta), 4),
            'max_drawdown': round(max_drawdown * 100, 2),
            'best_day':     round(float(portfolio_return.max()) * 100, 2),
            'worst_day':    round(float(portfolio_return.min()) * 100, 2),
            'chart1': to_html(fig1),
            'chart2': to_html(fig2),
            'chart3': to_html(fig3),
            'chart4': to_html(fig4),
            'chart5': to_html(fig5),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
