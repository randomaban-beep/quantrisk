-- Daily drawdown by strategy, with running peaks and contiguous underwater windows.
WITH peaks AS (
    SELECT date, strategy, nav,
           max(nav) OVER (PARTITION BY strategy ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_peak
    FROM daily_nav
), underwater AS (
    SELECT *, nav / nullif(running_peak, 0) - 1 AS drawdown FROM peaks
)
SELECT date, strategy, drawdown,
       sum(CASE WHEN drawdown = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY strategy ORDER BY date) AS drawdown_window
FROM underwater ORDER BY strategy, date;
