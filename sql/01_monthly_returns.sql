-- Monthly net return pivot by strategy.
WITH monthly AS (
    SELECT date_trunc('month', date)::DATE AS month, strategy, product(1 + return_net) - 1 AS monthly_return
    FROM strategy_returns GROUP BY 1, 2
)
SELECT month,
       max(monthly_return) FILTER (WHERE strategy = 'equal_weight') AS equal_weight,
       max(monthly_return) FILTER (WHERE strategy = 'sixty_forty') AS sixty_forty,
       max(monthly_return) FILTER (WHERE strategy = 'inverse_vol') AS inverse_vol,
       max(monthly_return) FILTER (WHERE strategy = 'min_variance') AS min_variance,
       max(monthly_return) FILTER (WHERE strategy = 'max_sharpe') AS max_sharpe,
       max(monthly_return) FILTER (WHERE strategy = 'risk_parity') AS risk_parity,
       max(monthly_return) FILTER (WHERE strategy = 'hrp') AS hrp
FROM monthly GROUP BY month ORDER BY month;
