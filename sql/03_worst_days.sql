-- Ten worst net days per strategy and the largest absolute weighted asset contribution.
WITH contributions AS (
    SELECT r.date, r.strategy, r.return_net, w.asset,
           w.weight * a.asset_return AS contribution,
           row_number() OVER (PARTITION BY r.date, r.strategy ORDER BY abs(w.weight * a.asset_return) DESC) AS asset_rank
    FROM strategy_returns r
    JOIN daily_weights w USING (date, strategy)
    JOIN asset_returns a USING (date, asset)
    WHERE w.weight_type = 'drifted'
), dominant AS (
    SELECT * FROM contributions WHERE asset_rank = 1
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY strategy ORDER BY return_net) AS day_rank
    FROM dominant
)
SELECT date, strategy, return_net, asset AS dominant_asset, contribution
FROM ranked WHERE day_rank <= 10 ORDER BY strategy, day_rank;
