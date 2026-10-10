-- Exception counts and rates by year, model, confidence, and strategy.
SELECT year(date) AS year, strategy, model, confidence,
       count(*) FILTER (WHERE realized < -var) AS exceptions,
       count(*) * (1 - confidence) AS expected_exceptions,
       avg(CASE WHEN realized < -var THEN 1.0 ELSE 0.0 END) AS exception_rate
FROM var_forecasts GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4;
