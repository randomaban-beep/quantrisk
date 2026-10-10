# Methodology

## Portfolio construction

The engine uses simple daily returns and a 252-day annualization. Expected returns blend historical asset means with their cross-sectional mean. Covariance may be sample, Ledoit–Wolf shrinkage, or EWMA (decay 0.94). Equal weight and inverse volatility are direct rules; minimum variance minimizes `w'Σw`; maximum Sharpe uses bounded SLSQP; risk parity minimizes squared deviations of normalized variance contributions; HRP uses correlation distance `sqrt((1-ρ)/2)`, single linkage, quasi-diagonal ordering, and recursive inverse-variance allocation. Long-only weights are fully invested and capped at 40% when feasible.

Walk-forward signals use a rolling history through date *t*. The engine trades at the close after the configured lag, deducts costs from turnover, and only lets new holdings earn returns on later observations. Between trades, holdings drift with returns.

## Risk and validation

Historical simulation uses empirical lower-tail quantiles. Gaussian VaR uses EWMA covariance. Cornish–Fisher adjusts the normal quantile for skew and excess kurtosis. Filtered historical simulation standardizes portfolio returns by EWMA volatility before estimating the empirical tail. VaR is a positive loss. Kupiec tests unconditional exception frequency; Christoffersen tests exception independence; conditional coverage combines the likelihood ratios. Basel traffic-light zones use rolling 250-observation 99% exceptions.

Historical stress replays fixed latest weights. Hypothetical shocks are illustrative assumptions. Correlation stress raises off-diagonal correlations to a configured floor while retaining individual volatilities. Uncertainty intervals use the Politis–Romano stationary bootstrap; sensitivity changes one configuration factor at a time.

## References

- Markowitz (1952), “Portfolio Selection,” *Journal of Finance*.
- Ledoit and Wolf (2004), “Honey, I Shrunk the Sample Covariance Matrix,” *Journal of Portfolio Management*.
- Maillard, Roncalli, and Teïletche (2010), “The Properties of Equally Weighted Risk Contribution Portfolios.”
- López de Prado (2016), “Building Diversified Portfolios that Outperform Out of Sample.”
- Kupiec (1995), “Techniques for Verifying the Accuracy of Risk Measurement Models.”
- Christoffersen (1998), “Evaluating Interval Forecasts.”
- Basel Committee (1996), “Supervisory Framework for the Use of Backtesting in Conjunction with the Internal Models Approach to Market Risk Capital Requirements.”
- Politis and Romano (1994), “The Stationary Bootstrap.”
- Cornish and Fisher (1937), “Moments and Cumulants in the Specification of Distributions.”
- Barone-Adesi, Giannopoulos, and Vosper (1999), “VaR without Correlations for Nonlinear Portfolios.”
