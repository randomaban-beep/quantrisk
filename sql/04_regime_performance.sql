-- Calendar-year and named-regime net returns exported by the analytics stage.
SELECT strategy, period, return AS cumulative_return
FROM subperiods ORDER BY strategy, period;
