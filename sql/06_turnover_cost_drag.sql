-- Yearly turnover and transaction cost drag.
SELECT year(date) AS year, strategy,
       sum(turnover) AS total_turnover,
       sum(cost) AS total_cost_drag,
       avg(turnover) * 252 AS annualized_turnover
FROM turnover_costs GROUP BY 1, 2 ORDER BY 1, 2;
